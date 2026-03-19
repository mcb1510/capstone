import requests
import os
import time
import json
import re
import numpy as np
from difflib import SequenceMatcher  # for fuzzy name matching
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# FAISS is required for chunk-level retrieval
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

load_dotenv()
MODEL_NAME = "us.anthropic.claude-sonnet-4-20250514-v1:0"

# ============================
# Helper functions
# ============================

def _similarity(a: str, b: str) -> float:
    """Return a similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def _detect_list_query(text: str) -> bool:
    """Detect if the user is asking for a list of all faculty using regex-based intent detection."""
    q = text.lower()
    patterns = [
        r"(list|show|display|give|tell me).*(all|everyone|every).*(faculty|professors? )",
        r"(list|show|display).*(faculty|professors?)",
        r"(who are|what are).*(all|everyone).*(faculty|professors?)",
        r"(all|everyone).*(faculty|professors? )",
    ]
    return any(re.search(pattern, q) for pattern in patterns)

def _detect_list_with_research_query(text: str) -> bool:
    """Detect if user wants faculty list WITH research areas."""
    q = text.lower()
    return bool(re.search(r"(list|show).*(faculty|professors? ).*(research|areas?|interests?)", q))

# HELPER CLASS FOR QUERY EXPANSION
class QueryProcessor:
    """Expands queries with domain-specific synonyms"""

    def __init__(self):
        self.research_synonyms = {
            "ai": ["artificial intelligence", "machine learning", "deep learning", "neural networks"],
            "ml": ["machine learning", "deep learning", "statistical learning"],
            "security": ["cybersecurity", "privacy", "cryptography", "network security"],
            "hci": ["human computer interaction", "user experience", "interface design", "usability"],
            "nlp": ["natural language processing", "computational linguistics", "text mining"],
            "cv": ["computer vision", "image processing", "pattern recognition"],
            "systems": ["distributed systems", "operating systems", "cloud computing", "parallel computing"],
            "blockchain": ["distributed ledger", "cryptocurrency", "consensus protocols"],
        }

    def expand_query(self, query: str) -> str:
        """Expand query with synonyms."""
        query_lower = query.lower()
        expanded_terms = []

        for keyword, synonyms in self.research_synonyms.items():
            if keyword in query_lower:
                expanded_terms.extend(synonyms)

        if expanded_terms:
            return f"{query} {' '.join(set(expanded_terms))}"
        return query


class ResponseEngine:
    """
    Response engine using Boise State API
    plus retrieval-augmented generation (RAG)
    over BSU CS faculty profiles.

    This version implements CHUNK-LEVEL retrieval with:
    - faculty_chunks.faiss (FAISS index over chunk embeddings)
    - chunks_meta.json (chunk_id -> {faculty_id, chunk_type, chunk_text})
    - faculty_metadata.json (faculty_id -> contact info, links)
    """

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

            hits.append(
                {
                    "chunk_id": chunk_id,
                    "faculty_id": faculty_id,
                    "faculty_name": faculty_name,
                    "chunk_type": meta.get("chunk_type", ""),
                    "chunk_text": meta.get("chunk_text", ""),
                    "score": float(score),
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

    def _list_all_faculty_text(self):
        """Return a human readable list of all faculty names."""
        if not self.faculty_ids:
            return "I do not have any faculty data loaded right now."

        self.conversation_memory = {"last_query": None, "last_retrieved": None}

        lines = [f"- {name}" for name in self.faculty_ids]
        return (
            "Here is the list of CS faculty I know about:\n\n"
            + "\n".join(lines)
            + "\n\nYou can ask me about any specific person, or tell me your interests and I will recommend a few advisors."
        )

    def _faculty_research_areas_from_chunks(self, faculty_id: str) -> str:
        areas = []
        if not self.chunks_meta:
            return ""
        for _, meta in self.chunks_meta.items():
            if str(meta.get("faculty_id")) == str(faculty_id) and meta.get("chunk_type") == "research_areas":
                text = (meta.get("chunk_text") or "").strip()
                if text:
                    areas.append(text)
        return "; ".join(areas)

    def _list_all_faculty_with_research(self):
        """Return faculty list with research areas based on research_areas chunks."""
        if not self.faculty_ids or not self.faculty_meta:
            return "I do not have any faculty data loaded right now."

        self.conversation_memory = {"last_query": None, "last_retrieved": None}

        lines = []
        for fid in sorted(self.faculty_meta.keys(), key=lambda x: int(x)):
            name = self.faculty_meta[fid].get("name", f"faculty_{fid}")
            research = self._faculty_research_areas_from_chunks(str(fid)) or "Research areas not listed"
            lines.append(f"• **{name}**: {research}")

        return (
            "Here is the list of CS faculty with their research areas:\n\n"
            + "\n\n".join(lines)
            + "\n\nAsk me about any specific professor for more details!"
        )

    def _build_faculty_profile_from_chunks(self, faculty_id: str) -> str:
        """
        Build a lightweight profile text for a faculty member from chunk metadata.
        This replaces the old faculty_texts.json profile dump.
        """
        fmeta = self.faculty_meta.get(str(faculty_id), {})
        name = fmeta.get("name", f"faculty_{faculty_id}")
        position = fmeta.get("position", "")
        email = fmeta.get("email", "")
        office = fmeta.get("office", "")
        profile_link = fmeta.get("profile_link", "")
        scholar = fmeta.get("google_scholar_link", "")

        by_type = {}
        for _, meta in (self.chunks_meta or {}).items():
            if str(meta.get("faculty_id")) != str(faculty_id):
                continue
            ctype = meta.get("chunk_type", "other")
            text = (meta.get("chunk_text") or "").strip()
            if not text:
                continue
            by_type.setdefault(ctype, []).append(text)

        parts = []
        parts.append(f"Name: {name}")
        if position:
            parts.append(f"Position: {position}")
        if email:
            parts.append(f"Email: {email}")
        if office:
            parts.append(f"Office: {office}")
        if profile_link:
            parts.append(f"Profile: {profile_link}")
        if scholar:
            parts.append(f"Google Scholar: {scholar}")

        # Add chunk content
        if "research_areas" in by_type:
            parts.append("Research Areas: " + "; ".join(by_type["research_areas"]))
        if "keywords" in by_type:
            parts.append("Keywords: " + "; ".join(by_type["keywords"]))
        if "publications" in by_type:
            # keep short
            pubs = by_type["publications"][:5]
            parts.append("Selected Publications: " + " | ".join(pubs))

        # If you add more chunk types later, they will still appear
        other_types = [t for t in by_type.keys() if t not in {"research_areas", "keywords", "publications"}]
        for t in sorted(other_types):
            parts.append(f"{t}: " + " | ".join(by_type[t][:5]))

        return "\n".join(parts)

    def _answer_for_specific_faculty(self, faculty_name, history=None):
        """
        Build a focused prompt for one matched faculty member.
        """
        if not self.faculty_ids or not self.faculty_meta:
            return "I could not load the faculty profiles right now."

        # Find best match by name similarity against faculty metadata
        best_fid = None
        best_score = 0.0
        for fid, meta in self.faculty_meta.items():
            name = meta.get("name", "")
            s = _similarity(faculty_name, name)
            if s > best_score:
                best_score = s
                best_fid = fid

        if best_fid is None or best_score < 0.65:
            return "I could not find that faculty in my profiles."

        profile = self._build_faculty_profile_from_chunks(str(best_fid))
        canonical_name = self.faculty_meta[str(best_fid)].get("name", faculty_name)

        # Store memory for follow-up questions
        self.conversation_memory["last_query"] = canonical_name
        self.conversation_memory["last_retrieved"] = [
            {
                "faculty_id": str(best_fid),
                "faculty_name": canonical_name,
                "profile_text": profile,
            }
        ]

        prompt = f"""
You are the AI Graduate Advisor for Boise State University.

The user is asking about: {canonical_name}

FACULTY PROFILE:
{profile}

INSTRUCTIONS:
1. Give a concise but rich summary of this professor's research areas.
2. Explain what makes their research interesting or impactful (3 to 4 sentences).
3. Describe what background, skills, and interests graduate students typically need to work with this professor (2 to 3 sentences).
4. Include all available contact information: email, office location, and Google Scholar link.
5. Keep the answer helpful, direct, and focused.
6. Be specific about their research. Use the actual topics from their profile.
7. Avoid repeating boilerplate language.
8. Always ask at the end if there is anything else you can help with (one short question).

TONE:
- Informative and professional but conversational
- Helpful and practical for students making decisions
"""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Tell me about {canonical_name} as a potential advisor for me."},
        ]

        if history:
            for msg in history[-3:]:
                messages.insert(1, {"role": msg["role"], "content": msg["content"]})

        return self._query_bsu(messages, max_tokens=600)

    def _answer_followup_fact(self, faculty_name, user_query, history=None):
        """
        Extract specific factual information from the stored faculty profile.
        """
        if not self.faculty_meta:
            return "I could not load the faculty profiles right now."

        # Find faculty id by matching the stored name
        best_fid = None
        best_score = 0.0
        for fid, meta in self.faculty_meta.items():
            name = meta.get("name", "")
            s = _similarity(faculty_name, name)
            if s > best_score:
                best_score = s
                best_fid = fid

        if best_fid is None or best_score < 0.65:
            return "I couldn't find that faculty member anymore."

        profile = self._build_faculty_profile_from_chunks(str(best_fid))

        conversation_context = ""
        if history and len(history) > 0:
            recent_messages = history[-6:]
            conversation_context = "RECENT CONVERSATION:\n"
            for msg in recent_messages:
                role = "USER" if msg.get("role") == "user" else "ASSISTANT"
                content = msg.get("content", "")
                if len(content) > 300:
                    content = content[:300] + "..."
                conversation_context += f"{role}: {content}\n\n"

        prompt = f"""
You are the BSU Graduate Advisor AI Assistant. You are helping a student learn about Professor {faculty_name}.

{conversation_context}

FACULTY PROFILE:
{profile}

CURRENT USER INPUT:
{user_query}

INSTRUCTIONS:
- If the user said "yes", "sure", "ok", or similar, look at the RECENT CONVERSATION to see what you offered.
- Provide the specific information that was offered in your previous message.
- If you cannot determine what they want from context, ask for clarification.
- Answer directly in complete sentences.
- After answering, offer to help further with one short follow-up question.
- If the information is not in the profile, say so politely.
"""
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_query},
        ]
        return self._query_bsu(messages, max_tokens=200)

    def _is_followup(self, query: str) -> bool:
        q = query.lower()
        for name in (self.faculty_ids or []):
            if name.lower() in q:
                return False
            sim = _similarity(q, name.lower())
            if sim > 0.65:
                return False

        mem_exists = self.conversation_memory.get("last_retrieved") is not None
        return mem_exists

    # =================================================================
    # Query classification and concept definition (unchanged)
    # =================================================================

    def classify_query_type(self, query):
        """
        Classify the user query into one of:
        - followup_person
        - general_concept
        - new_professor
        """
        system_prompt = """
You are a query classifier. Classify the user's question into ONE of these:

1. followup_person:
- The question refers to the previously discussed professor.
- Includes pronouns like he, him, his, she, her, they, them.
- Includes questions about their office, email, research areas, advising, etc.

2. general_concept:
- The question asks about a research field, definition, concept, method,
  technique, or career possibilities (for example: "what is X?").

3. new_professor:
- The question is asking about a different professor OR requesting new advisor recommendations.

Respond with ONLY: followup_person, general_concept, or new_professor.
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        result = self._query_bsu(messages, max_tokens=5).lower().strip()

        if "follow" in result:
            return "followup_person"
        if "concept" in result:
            return "general_concept"
        if "professor" in result or "new" in result:
            return "new_professor"

        return "general_concept"

    def _answer_concept_definition(self, query):
        prompt = f"""
You are an AI assistant. Provide a clear explanation for the research concept or topic the user is asking about.

Requirements:
- Give a correct 2 to 4 sentence definition.
- Use examples relevant to Computer Science.
- If appropriate, mention what careers or research areas use this concept.
- Always ask if there is anything else you can help with (one short question) and explicitly offer to recommend BSU CS faculty who match the topic of the question.

USER QUESTION:
{query}
"""
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ]
        return self._query_bsu(messages, max_tokens=300)

    # =================================================================
    # LOW LEVEL BSU CALL
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

    def generate_rag_answer(self, user_query, history=None, top_k=5):
        """
        RAG mode:
        1) List-all queries and name matches.
        2) Otherwise retrieve chunks, aggregate to faculty, inject evidence context.
        """
        if _detect_list_with_research_query(user_query) and self.faculty_ids:
            return self._list_all_faculty_with_research()

        if _detect_list_query(user_query) and self.faculty_ids:
            return self._list_all_faculty_text()

        # Direct faculty name mentions
        if self.faculty_ids:
            q = user_query.lower()
            query_tokens = q.split()

            for name in self.faculty_ids:
                name_tokens = name.lower().split()

                token_matches = sum(1 for qt in query_tokens for nt in name_tokens if qt == nt)
                if token_matches >= 2:
                    print("TOKEN DIRECT MATCH:", name)
                    return self._answer_for_specific_faculty(name, history=history)

                fuzzy_matches = sum(1 for qt in query_tokens for nt in name_tokens if _similarity(qt, nt) > 0.70)
                if fuzzy_matches >= 2:
                    print("TOKEN FUZZY MATCH:", name)
                    return self._answer_for_specific_faculty(name, history=history)

        # Affirmative or negative follow-ups
        affirmative_patterns = [
            r"^(yes|yeah|yep|yup|sure|ok|okay|alright|please|yes please|sure thing)[.!?]?$",
            r"^(yes|yeah|sure),?\s+(tell me|show me|give me|send me|what about)\b",
            r"^tell me more[.!?]?$",
            r"^show me more[.!?]?$",
            r"^more[.!?]?$",
            r"^(that would be|that\'d be|sounds)\s+(great|good|helpful|perfect|nice)[.!?]?$",
            r"^(go ahead|please do|i\'m interested)[.!?]?$",
        ]

        negative_patterns = [
            r"^(no|nope|nah|no thanks|no thank you)\.?!?$",
            r"^(that\'s all|that\'s it|i\'m good|i\'m all set)\.?$",
            r"^(nothing else|nothing more)\.?$",
            r"^(i\'m done|all done)\.?$",
        ]

        query_lower = user_query.lower().strip()
        is_affirmative = any(re.match(pattern, query_lower) for pattern in affirmative_patterns)
        is_negative = any(re.match(pattern, query_lower) for pattern in negative_patterns)

        if is_affirmative:
            last = self.conversation_memory.get("last_retrieved")
            if last and len(last) > 0:
                faculty_name = last[0].get("faculty_name") or last[0].get("name") or ""
                if faculty_name:
                    return self._answer_followup_fact(faculty_name, user_query, history=history)

        if is_negative:
            return (
                "No problem! Feel free to ask me about other faculty members, research areas, or anything else about the BSU CS graduate program. "
                "How else can I help you?"
            )

        # Query classification
        query_type = self.classify_query_type(user_query.lower())

        last = self.conversation_memory.get("last_retrieved")
        last_prof = None
        if last and len(last) > 0:
            last_prof = last[0].get("faculty_name") or last[0].get("name")

        if query_type == "followup_person" and last_prof:
            return self._answer_followup_fact(last_prof, user_query, history=history)

        if query_type == "general_concept":
            return self._answer_concept_definition(user_query)

        # Enhanced retrieval: chunks -> aggregated faculty
        retrieved = self.retrieve_faculty(user_query, top_k=top_k, top_k_chunks=30, per_faculty_cap=3)
        self.conversation_memory["last_query"] = user_query
        self.conversation_memory["last_retrieved"] = retrieved

        if not retrieved:
            return (
                "I could not match your question to any specific faculty profiles. "
                "Try telling me your research interests, for example: "
                "\"I am interested in AI and machine learning\" or "
                "\"I want to work on cybersecurity and privacy\"."
            )

        # Debug quick check (top chunk evidence from the top faculty result)
        top_ev = retrieved[0].get("evidence_chunks", []) if retrieved else []
        print("TOP EVIDENCE:", [(c.get("chunk_type"), round(c.get("score", 0.0), 3)) for c in top_ev[:5]])

        context_blocks = []
        for i, r in enumerate(retrieved, start=1):
            ev_lines = []
            for c in r.get("evidence_chunks", []):
                ev_lines.append(f"- ({c.get('chunk_type','')}, {c.get('score',0.0):.3f}) {c.get('chunk_text','')}")
            block = (
                f"FACULTY MATCH {i}:\n"
                f"Name: {r.get('faculty_name','')}\n"
                f"Aggregated score: {r.get('score',0.0):.3f}\n"
                f"Email: {r.get('email','')}\n"
                f"Office: {r.get('office','')}\n"
                f"Profile: {r.get('profile_link','')}\n"
                f"Scholar: {r.get('google_scholar_link','')}\n"
                f"Evidence:\n" + "\n".join(ev_lines) + "\n"
            )
            context_blocks.append(block)

        faculty_context = "\n---\n".join(context_blocks)

        rag_system_prompt = (
            "You are the BSU Graduate Advisor AI Assistant for Computer Science students at Boise State University.\n\n"
            "You are connected to a factual database of BSU CS faculty chunk evidence.\n"
            "Below you are given the top retrieved faculty candidates and the evidence chunks that matched the student's question.\n\n"
            "=== FACULTY CONTEXT START ===\n"
            f"{faculty_context}\n"
            "=== FACULTY CONTEXT END ===\n\n"
            "Instructions:\n"
            "- When recommending advisors, rely ONLY on the evidence in the faculty context.\n"
            "- Recommend 1 to 3 specific faculty that best match the student's interests.\n"
            "- Briefly explain why each recommended faculty member is a good match.\n"
            "- Do not invent research areas, publications, or contact info.\n"
            "- Do not ask unnecessary clarifying questions. Make the best recommendation with the information you have.\n"
            "- Keep answers concise (2 to 4 sentences).\n"
        )

        messages = [{"role": "system", "content": rag_system_prompt}]

        if history:
            for msg in history[-4:]:
                if msg.get("role") in ("user", "assistant"):
                    messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_query})

        return self._query_bsu(messages, max_tokens=800)
