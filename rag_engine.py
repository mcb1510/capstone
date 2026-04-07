from rag_core import ResponseEngineCore
from rag_faculty import ResponseEngineFaculty
from rag_shared import _detect_list_query, _detect_list_with_research_query
import re

class ResponseEngine(ResponseEngineCore, ResponseEngineFaculty):
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
        direct_name_match = self._find_best_faculty_name_match(user_query)
        if direct_name_match:
            print("DIRECT NAME MATCH:", direct_name_match)
            if self.is_fact_query(user_query):
                return self._answer_followup_fact(direct_name_match, user_query, history=history)
            return self._answer_for_specific_faculty(direct_name_match, history=history)

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
            pending_concept = self.conversation_memory.get("pending_concept_query")
            if pending_concept:
                retrieved = self.retrieve_faculty(pending_concept, top_k=top_k, top_k_chunks=30, per_faculty_cap=3)
                self.conversation_memory["pending_concept_query"] = None
                self.conversation_memory["last_query"] = pending_concept
                self.conversation_memory["last_retrieved"] = retrieved

                if not retrieved:
                    return "I could not find strong faculty matches for that topic from the available profiles."

                faculty_context = ""
                for i, item in enumerate(retrieved[:3], start=1):
                    faculty_context += f"""
FACULTY {i}
Name: {item.get('faculty_name', '')}
Email: {item.get('email', '')}
Office: {item.get('office', '')}
Profile: {item.get('profile_link', '')}
Google Scholar: {item.get('google_scholar_link', '')}
Relevant evidence:
"""
                    for ev in item.get("evidence_chunks", [])[:3]:
                        faculty_context += (
                            f"- [{ev.get('chunk_type', 'chunk')}] {ev.get('chunk_text', '')[:900]}\n"
                        )

                prompt = f"""
You are the BSU Graduate Advisor AI Assistant.

The user previously asked about this research topic:
{pending_concept}

They then replied with an affirmative answer like "yes", meaning they want BSU CS faculty recommendations for that topic.

Use ONLY the retrieved faculty evidence below.

{faculty_context}

INSTRUCTIONS:
- Recommend 1 to 3 faculty members who best match the topic.
- Explain why each recommendation matches the topic using the retrieved evidence.
- Be direct and concise.
- Do not invent details not supported by the evidence.
- End with one short follow-up question.
"""
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": pending_concept},
                ]
                return self._query_bsu(messages, max_tokens=350)

            offer_fulfillment = self._answer_affirmative_to_offer(user_query, history=history)
            if offer_fulfillment:
                return offer_fulfillment

            active_name = self.conversation_memory.get("active_faculty_name")
            if active_name:
                return self._answer_followup_fact(active_name, user_query, history=history)

            last = self.conversation_memory.get("last_retrieved")
            if last and len(last) > 0:
                fallback_name = last[0].get("faculty_name") or last[0].get("name") or ""
                if fallback_name:
                    return self._answer_followup_fact(fallback_name, user_query, history=history)

        if is_negative:
            return (
                "No problem! Feel free to ask me about other faculty members, research areas, or anything else about the BSU CS graduate program. "
                "How else can I help you?"
            )

        if self._should_route_to_followup(user_query, history=history):
            followup_name = self._resolve_followup_faculty_name(user_query)
            if followup_name:
                return self._answer_followup_fact(followup_name, user_query, history=history)

        # Query classification
        query_type = self.classify_query_type(user_query.lower())

        last = self.conversation_memory.get("last_retrieved")
        last_prof = None
        if last and len(last) > 0:
            last_prof = last[0].get("faculty_name") or last[0].get("name")

        if query_type == "followup_person" and last_prof:
            return self._answer_followup_fact(last_prof, user_query, history=history)

        if query_type == "general_concept":
            self.conversation_memory["pending_concept_query"] = user_query
            self.conversation_memory["last_query"] = user_query
            self.conversation_memory["last_retrieved"] = None
            self.conversation_memory["active_faculty_name"] = None
            return self._answer_concept_definition(user_query)

        # Enhanced retrieval: chunks -> aggregated faculty
        retrieved = self.retrieve_faculty(user_query, top_k=top_k, top_k_chunks=30, per_faculty_cap=3)
        self.conversation_memory["pending_concept_query"] = None
        self.conversation_memory["last_query"] = user_query
        self.conversation_memory["last_retrieved"] = retrieved
        self.conversation_memory["active_faculty_name"] = (retrieved[0].get("faculty_name") if retrieved else None)

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
