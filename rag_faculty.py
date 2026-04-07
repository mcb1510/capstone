from rag_shared import _similarity
import re

class ResponseEngineFaculty:
    def is_fact_query(self, query):
            prompt = f"""
        You are a strict classifier.

        You MUST return EXACTLY one of these two tokens:
        fact
        profile

        No punctuation.
        No explanation.
        No extra words.

        If the query asks for a specific detail about a professor (location, contact info, etc.), return:
        fact

        Otherwise return:
        profile

        Query: {query}
        """

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ]

            result = self._query_bsu(messages, max_tokens=8).strip().lower()

            print("CLASSIFIER RAW:", repr(result))  # debug

            first_line = result.splitlines()[0].strip() if result else ""
            if re.search(r"\bfact\b", first_line):
                return True
            if re.search(r"\bprofile\b", first_line):
                return False

            q = query.lower()
            fact_terms = ["email", "office", "where", "contact", "scholar", "phone", "location"]
            return any(term in q for term in fact_terms)

    def _find_best_faculty_name_match(self, user_query: str):
        """Find best faculty name match from query with conservative acceptance criteria."""
        if not self.faculty_ids:
            return None

        q = user_query.lower()
        query_tokens = [t for t in re.findall(r"[a-z0-9]+", q) if t]
        query_token_set = set(query_tokens)

        stop_tokens = {
            "tell", "me", "about", "who", "is", "the", "a", "an", "and", "of", "for", "with",
            "to", "on", "in", "at", "from", "please", "advisor", "professor", "prof", "dr",
        }
        query_content_tokens = {t for t in query_token_set if t not in stop_tokens}

        # Fast-path: if query contains a single unique faculty token (e.g., first name only),
        # resolve directly to that faculty.
        token_to_faculty = {}
        for name in self.faculty_ids:
            for token in set(re.findall(r"[a-z0-9]+", name.lower())):
                if token not in token_to_faculty:
                    token_to_faculty[token] = name
                else:
                    token_to_faculty[token] = None

        unique_query_matches = [
            token_to_faculty[t]
            for t in query_content_tokens
            if t in token_to_faculty and token_to_faculty[t] is not None
        ]
        if len(unique_query_matches) == 1:
            return unique_query_matches[0]

        best_name = None
        best_score = -1.0
        best_exact_overlap = 0
        best_fuzzy_overlap = 0
        best_name_similarity = 0.0
        best_full_in_query = False

        for name in self.faculty_ids:
            name_lower = name.lower()
            name_tokens = set(re.findall(r"[a-z0-9]+", name_lower))
            full_in_query = name_lower in q

            exact_overlap = len(query_content_tokens.intersection(name_tokens))

            fuzzy_overlap = 0
            for qt in query_content_tokens:
                for nt in name_tokens:
                    if _similarity(qt, nt) >= 0.88:
                        fuzzy_overlap += 1
                        break

            name_similarity = _similarity(q, name_lower)
            score = exact_overlap * 2.0 + fuzzy_overlap * 1.0 + (2.0 if full_in_query else 0.0) + name_similarity

            if score > best_score:
                best_score = score
                best_name = name
                best_exact_overlap = exact_overlap
                best_fuzzy_overlap = fuzzy_overlap
                best_name_similarity = name_similarity
                best_full_in_query = full_in_query

        if not best_name:
            return None

        if best_full_in_query:
            return best_name
        if best_exact_overlap >= 2:
            return best_name
        if best_exact_overlap >= 1 and len(query_content_tokens) == 1:
            return best_name
        if best_fuzzy_overlap >= 2:
            return best_name
        if best_name_similarity >= 0.82:
            return best_name

        return None

    def _resolve_followup_faculty_name(self, user_query: str):
        """Resolve which previously retrieved faculty a follow-up likely refers to."""
        last = self.conversation_memory.get("last_retrieved") or []
        if not last:
            return None

        q = user_query.lower()
        query_tokens = set(re.findall(r"[a-z0-9]+", q))

        best_name = None
        best_overlap = -1

        for item in last:
            candidate = item.get("faculty_name") or item.get("name") or ""
            if not candidate:
                continue

            cand_lower = candidate.lower()
            if cand_lower in q:
                return candidate

            cand_tokens = set(re.findall(r"[a-z0-9]+", cand_lower))
            overlap = len(query_tokens.intersection(cand_tokens))
            if overlap > best_overlap:
                best_overlap = overlap
                best_name = candidate

        if best_overlap >= 2 and best_name:
            return best_name

        active_name = self.conversation_memory.get("active_faculty_name")
        if active_name:
            return active_name

        return (last[0].get("faculty_name") or last[0].get("name") or None)

    def _list_all_faculty_text(self):
        """Return a human readable list of all faculty names."""
        if not self.faculty_ids:
            return "I do not have any faculty data loaded right now."

        self.conversation_memory = {
            "last_query": None,
            "last_retrieved": None,
            "pending_concept_query": None,
            "active_faculty_name": None,
        }

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

        self.conversation_memory = {
            "last_query": None,
            "last_retrieved": None,
            "pending_concept_query": None,
            "active_faculty_name": None,
        }

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
        self.conversation_memory["active_faculty_name"] = canonical_name

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

        messages = [{"role": "system", "content": prompt}]

        if history:
            for msg in history[-3:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": f"Tell me about {canonical_name} as a potential advisor for me."})

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

        canonical_name = self.faculty_meta[str(best_fid)].get("name", faculty_name)
        self.conversation_memory["active_faculty_name"] = canonical_name

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
        return self._query_bsu(messages, max_tokens=450)

    def _last_assistant_message(self, history=None):
        if not history:
            return ""
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                return (msg.get("content") or "").strip()
        return ""

    def _is_offer_question(self, text: str) -> bool:
        t = (text or "").lower()
        if not t:
            return False
        patterns = [
            r"would you like",
            r"do you want",
            r"want me to",
            r"should i",
            r"can i help",
            r"would it help if",
            r"i can help",
        ]
        return any(re.search(p, t) for p in patterns)

    def _answer_affirmative_to_offer(self, user_query, history=None):
        """
        If user gives an affirmative reply (yes/ok/sure), fulfill the most recent
        assistant offer question directly.
        """
        last_assistant = self._last_assistant_message(history)
        if not last_assistant or not self._is_offer_question(last_assistant):
            return None

        target_faculty = self.conversation_memory.get("active_faculty_name")
        if not target_faculty and history:
            target_faculty = self._resolve_followup_faculty_name(user_query)

        profile = ""
        if target_faculty and self.faculty_meta:
            best_fid = None
            best_score = 0.0
            for fid, meta in self.faculty_meta.items():
                name = meta.get("name", "")
                s = _similarity(target_faculty, name)
                if s > best_score:
                    best_score = s
                    best_fid = fid

            if best_fid is not None and best_score >= 0.65:
                canonical_name = self.faculty_meta[str(best_fid)].get("name", target_faculty)
                self.conversation_memory["active_faculty_name"] = canonical_name
                profile = self._build_faculty_profile_from_chunks(str(best_fid))
                target_faculty = canonical_name

        conversation_context = ""
        if history:
            recent_messages = history[-8:]
            conversation_context = "RECENT CONVERSATION:\n"
            for msg in recent_messages:
                role = "USER" if msg.get("role") == "user" else "ASSISTANT"
                content = msg.get("content", "")
                if len(content) > 400:
                    content = content[:400] + "..."
                conversation_context += f"{role}: {content}\n\n"

        prompt = f"""
You are the BSU Graduate Advisor AI Assistant.

The user replied affirmatively (e.g., "yes") to your most recent offer.

MOST RECENT ASSISTANT OFFER:
{last_assistant}

CURRENT USER INPUT:
{user_query}

{conversation_context}

{f"FACULTY PROFILE:\n{profile}" if profile else ""}

INSTRUCTIONS:
- Fulfill exactly what you offered in the MOST RECENT ASSISTANT OFFER.
- Do NOT repeat a long professor summary unless needed for the requested deliverable.
- If the offer was to prepare questions for outreach, provide 6 to 8 concrete, specific questions.
- Keep the answer practical and directly usable.
- If the requested deliverable is unclear even with context, ask one concise clarification.
"""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_query},
        ]
        return self._query_bsu(messages, max_tokens=500)

    def _is_followup(self, query: str) -> bool:
        q = query.lower()

        new_search_markers = [
            "recommend", "suggest", "looking for", "interested in", "find advisor", "new professor",
            "who should", "which professor", "faculty for",
        ]
        if any(marker in q for marker in new_search_markers):
            return False

        for name in (self.faculty_ids or []):
            if name.lower() in q:
                return False
            sim = _similarity(q, name.lower())
            if sim > 0.65:
                return False

        mem_exists = self.conversation_memory.get("last_retrieved") is not None
        if not mem_exists:
            return False

        short_followup = len(q.split()) <= 8
        followup_markers = {"he", "she", "they", "him", "her", "their", "his", "hers", "it", "that", "this"}
        contains_followup_marker = any(tok in followup_markers for tok in re.findall(r"[a-z0-9]+", q))
        return short_followup or contains_followup_marker

    def _should_route_to_followup(self, user_query: str, history=None) -> bool:
        """
        Model-based routing to decide whether the current query should be treated
        as a follow-up to the current professor context.
        """
        if self.conversation_memory.get("last_retrieved") is None:
            return False

        active_name = self.conversation_memory.get("active_faculty_name") or ""

        conversation_context = ""
        if history:
            recent = history[-8:]
            for msg in recent:
                role = msg.get("role", "user").upper()
                content = (msg.get("content") or "").strip()
                if len(content) > 250:
                    content = content[:250] + "..."
                conversation_context += f"{role}: {content}\n"

        prompt = f"""
You are an intent router.

Decide if the CURRENT USER QUERY should be handled as FOLLOWUP_PERSON
or NEW_SEARCH.

Return ONLY one token:
FOLLOWUP_PERSON
NEW_SEARCH

Use conversation context and active professor name if available.
If the user is asking broadly about faculty for a topic, choose NEW_SEARCH.
If the user is asking about the currently discussed professor, choose FOLLOWUP_PERSON.

Active professor: {active_name}

Conversation:
{conversation_context}

CURRENT USER QUERY:
{user_query}
"""

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_query},
        ]

        result = self._query_bsu(messages, max_tokens=12).strip().upper()
        first_line = result.splitlines()[0].strip() if result else ""
        if "FOLLOWUP_PERSON" in first_line:
            return True
        if "NEW_SEARCH" in first_line:
            return False

        return self._is_followup(user_query)

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

        result = self._query_bsu(messages, max_tokens=12).lower().strip()
        first_line = result.splitlines()[0].strip() if result else ""

        if "followup_person" in first_line:
            return "followup_person"
        if "general_concept" in first_line:
            return "general_concept"
        if "new_professor" in first_line:
            return "new_professor"

        q = query.lower()
        if any(k in q for k in ["what is", "define", "explain", "difference between", "how does"]):
            return "general_concept"
        if any(k in q for k in ["he ", "she ", "they ", "his ", "her ", "their ", "email", "office"]):
            return "followup_person"
        return "new_professor"

    def _answer_concept_definition(self, query):
        prompt = f"""
You are an AI assistant. Provide a clear explanation for the research concept or topic the user is asking about.

Requirements:
- Give a correct 2 to 4 sentence definition.
- Use examples relevant to Computer Science.
- If appropriate, mention what careers or research areas use this concept.
- Always ask if something like if they want you to recommend BSU CS faculty who match the topic of the question.

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

