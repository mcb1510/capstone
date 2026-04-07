from rag_shared import *

class ResponseEngineCore:
    def __init__(self):
        """Initialize Boise State API connection and RAG resources."""

        # ---------- LLM setup ----------
        self.api_key = os.getenv("BSU_API_KEY", "")
        if not self.api_key:
            print("WARNING: No BSU_API_KEY found!")
            raise ValueError("BSU_API_KEY required in .env file")

        self.api_url = "https://api.boisestate.ai/chat/api-converse"
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }
        self.model = MODEL_NAME

        # General persona (used for non-RAG answers)
        self.system_prompt = (
            "You are the BSU Graduate Advisor AI Assistant for Computer Science "
            "students at Boise State University.\n\n"
            "Your role:\n"
            "- Help students find suitable research advisors based on their interests, skills, and goals\n"
            "- Provide information about faculty research areas and general availability\n"
            "- Guide students through the advisor selection process\n"
            "- Answer questions about BSU CS graduate programs\n"
            "- Be direct and concise (2 to 4 sentences)\n"
            "- Only ask a clarifying question if the student's request is genuinely ambiguous\n"
            "- When possible, make the best recommendation from available information instead of asking many follow up questions\n\n"
            "When you are provided with faculty data in the context, you MUST rely on that "
            "data and not invent additional details."
        )

        print(f"Boise State API initialized with modelId={self.model}")

        # ---------- RAG resources ----------
        self._load_rag_resources()

        # Query expansion
        self.query_processor = QueryProcessor()

        # Conversation memory for follow-ups
        self.conversation_memory = {
            "last_query": None,
            "last_retrieved": None,  # list of aggregated faculty dicts
            "pending_concept_query": None,
            "active_faculty_name": None,
        }

    def _messages_to_text(self, messages):
        """
        Convert OpenAI-style chat messages to one text prompt because
        Boise State /api-converse takes a single 'message' string.
        """
        parts = []
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            parts.append(f"{role}:\n{content}")
        return "\n\n".join(parts)

    # =================================================================
    # RAG INITIALIZATION
    # =================================================================

    def _load_rag_resources(self):
        """
        Load chunk-level retrieval resources.

        Expects these files in the current working directory:
        - faculty_chunks.faiss
        - chunks_meta.json
        - faculty_metadata.json
        """
        try:
            if faiss is None:
                raise ImportError(
                    "faiss is not installed. Install with: pip install faiss-cpu "
                    "or conda install -c conda-forge faiss-cpu"
                )

            print("[RAG] Loading FAISS chunk index and metadata...")

            self.faiss_index = faiss.read_index("faculty_chunks.faiss")
            self.faiss_metric_type = getattr(self.faiss_index, "metric_type", None)

            with open("chunks_meta.json", "r", encoding="utf-8") as f:
                # keys are strings: "0", "1", ...
                self.chunks_meta = json.load(f)

            with open("faculty_metadata.json", "r", encoding="utf-8") as f:
                # keys are strings: "0", "1", ...
                self.faculty_meta = json.load(f)

            # Convenience list of faculty names for name matching and list commands
            self.faculty_ids = [self.faculty_meta[k]["name"] for k in sorted(self.faculty_meta.keys(), key=lambda x: int(x))]
            self.faculty_ids_lower = [n.lower() for n in self.faculty_ids]

            print("[RAG] Loading BGE-large model for query encoding...")
            self.embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

            print(f"[RAG] Loaded {len(self.chunks_meta)} chunks.")
            print(f"[RAG] Loaded {len(self.faculty_meta)} faculty records.")
        except Exception as e:
            print(f"[RAG] WARNING: could not load RAG resources: {e}")
            self.faiss_index = None
            self.faiss_metric_type = None
            self.chunks_meta = None
            self.faculty_meta = None
            self.faculty_ids = None
            self.faculty_ids_lower = None
            self.embed_model = None

    # =================================================================
    # BASE LLM CALL (non-RAG)
    # =================================================================

    def generate_answer(self, user_query, history=None):
        """
        Plain LLM answer using only the static system_prompt.
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        if history:
            for msg in history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_query})

        return self._query_bsu(messages, max_tokens=400)

    def ask(self, user_query, history=None, use_rag=False):
        """
        If use_rag is True, use the RAG pipeline.
        Otherwise fall back to plain LLM.
        """
        if use_rag:
            return self.generate_rag_answer(user_query, history=history)
        return self.generate_answer(user_query, history=history)

    # =================================================================
    # RAG: CHUNK RETRIEVAL + AGGREGATION
    # =================================================================

    def _encode_query(self, query: str) -> np.ndarray:
        expanded_query = self.query_processor.expand_query(query)
        q_emb = self.embed_model.encode([expanded_query])[0].astype("float32")
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)
        return q_emb

    def retrieve_chunks(self, query: str, top_k_chunks: int = 30):
        """
        Retrieve top_k_chunks most relevant CHUNKS for a given query.
        Returns a list of dicts with:
        {chunk_id, faculty_id, faculty_name, chunk_type, chunk_text, score}
        """
        if self.embed_model is None or self.faiss_index is None:
            print("[RAG] Retrieval requested but RAG resources are not loaded.")
            return []

        q_emb = self._encode_query(query)
        D, I = self.faiss_index.search(np.expand_dims(q_emb, axis=0), top_k_chunks)

        hits = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx == -1:
                continue
            chunk_id = str(int(idx))
            meta = self.chunks_meta.get(chunk_id)
            if not meta:
                continue

            faculty_id = str(meta.get("faculty_id"))
            fmeta = self.faculty_meta.get(faculty_id, {})
            faculty_name = fmeta.get("name", f"faculty_{faculty_id}")

            normalized_score = float(score)
            if faiss is not None and self.faiss_metric_type == getattr(faiss, "METRIC_L2", -1):
                normalized_score = -normalized_score

            hits.append(
                {
                    "chunk_id": chunk_id,
                    "faculty_id": faculty_id,
                    "faculty_name": faculty_name,
                    "chunk_type": meta.get("chunk_type", ""),
                    "chunk_text": meta.get("chunk_text", ""),
                    "score": normalized_score,
                }
            )

        return hits

    def aggregate_faculty(self, chunk_hits, per_faculty_cap: int = 3):
        """
        Aggregate chunk hits into faculty-level scores.
        Aggregation rule:
        - for each faculty, take top per_faculty_cap chunk scores and sum them.
        """
        by_faculty = {}
        for h in chunk_hits:
            fid = h["faculty_id"]
            if fid not in by_faculty:
                by_faculty[fid] = {
                    "faculty_id": fid,
                    "faculty_name": h["faculty_name"],
                    "chunks": [],
                }
            by_faculty[fid]["chunks"].append(h)

        results = []
        for fid, item in by_faculty.items():
            chunks_sorted = sorted(item["chunks"], key=lambda x: x["score"], reverse=True)
            top_chunks = chunks_sorted[:per_faculty_cap]
            agg_score = sum(c["score"] for c in top_chunks)

            fmeta = self.faculty_meta.get(fid, {})
            results.append(
                {
                    "faculty_id": fid,
                    "faculty_name": item["faculty_name"],
                    "score": float(agg_score),
                    "evidence_chunks": top_chunks,
                    "email": fmeta.get("email", ""),
                    "office": fmeta.get("office", ""),
                    "profile_link": fmeta.get("profile_link", ""),
                    "google_scholar_link": fmeta.get("google_scholar_link", ""),
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    def retrieve_faculty(self, query: str, top_k: int = 5, top_k_chunks: int = 30, per_faculty_cap: int = 3):
        """
        Enhanced retrieval:
        - retrieve chunks via FAISS
        - aggregate into top_k faculty
        Returns a list of aggregated faculty dicts.
        """
        chunk_hits = self.retrieve_chunks(query, top_k_chunks=top_k_chunks)
        faculty_ranked = self.aggregate_faculty(chunk_hits, per_faculty_cap=per_faculty_cap)
        return faculty_ranked[:top_k]

    # =================================================================
    # Faculty listing and per-faculty profile building from chunks
    # =================================================================
    def _query_bsu(self, messages, max_retries=3, max_tokens=600):
        payload = {
            "message": self._messages_to_text(messages),
            "modelId": self.model,
            "temperature": 0.7,
            "maxTokens": max_tokens,
        }

        last_error = None

        for _attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30,
                )

                if response.status_code in (200, 201):
                    result = response.json()
                    return result.get("text", "").strip()

                if response.status_code == 401:
                    return "Authentication error. Check your BSU_API_KEY in the .env file."

                if response.status_code == 429:
                    time.sleep(2)
                    continue

                last_error = f"API Error {response.status_code}: {response.text}"

            except Exception as e:
                last_error = f"Request error: {e}"
                time.sleep(1)

        return last_error or "I'm having trouble connecting right now. Please try again."

    # =================================================================
    # RAG MODE: handlers + enhanced retrieval usage
    # =================================================================

