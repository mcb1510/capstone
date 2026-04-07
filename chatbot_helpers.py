from chatbot_config import st, json, GUIDED_QUESTIONS


# ==================== HELPERS ====================

def reset_to_welcome():
    st.session_state.mode = "welcome"
    st.session_state.messages = []
    st.session_state.guided_step = 0
    st.session_state.guided_answers = []
    if "generator" in st.session_state:
        st.session_state.generator.conversation_memory = {
            "last_query": None,
            "last_retrieved": None,
            "pending_concept_query": None,
            "active_faculty_name": None,
        }


def build_student_profile_text(answers):
    # Force the classifier into "new_professor" by making the intent explicit up front.
    # Keep the profile structured and keyword-rich, but avoid "TASK:" formatting that can confuse classification.

    return f"""Recommend 1 to 3 BSU CS faculty advisors based on the student profile below.
    First, write a short paragraph summarizing the student.
    Then recommend 1 to 3 advisors and explain why, grounded in the retrieved faculty evidence (chunks) only.

    STUDENT PROFILE
    Program/emphasis: {answers[0].strip()}
    Top interests: {answers[1].strip()}
    Courses liked: {answers[2].strip()}
    Bachelor's + undergrad projects: {answers[3].strip()}
    Strongest skills/tools: {answers[4].strip()}
    Do NOT want: {answers[5].strip()}
    Goal for grad school: {answers[6].strip()}
    """


def generate_guided_questions():
    prompt = """
Create exactly 7 guided interview questions for BSU CS graduate advisor matching.

Return ONLY valid JSON in this shape:
{"questions": ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]}

Question coverage (in order):
1) Program/emphasis
2) Top research interests
3) Courses liked
4) Bachelor's background + projects
5) Strongest skills/tools
6) Topics/project types to avoid
7) Goal for grad school

Keep questions concise, natural, and student-friendly.
"""

    raw = st.session_state.generator.ask(prompt, history=[], use_rag=False)

    try:
        parsed = json.loads(raw)
        questions = parsed.get("questions", []) if isinstance(parsed, dict) else []
        if isinstance(questions, list) and len(questions) == 7 and all(isinstance(q, str) and q.strip() for q in questions):
            return [q.strip() for q in questions]
    except Exception:
        pass

    return GUIDED_QUESTIONS.copy()


def assess_guided_answer(question, answer):
    prompt = f"""
You are validating a student's guided interview response.

Question:
{question}

Student answer:
{answer}

Return ONLY valid JSON with this exact schema:
{{
  "is_valid": true or false,
  "ack": "one short acknowledgement sentence",
  "feedback": "if invalid, one short instruction to improve; if valid, empty string"
}}

Rules:
- Mark invalid if answer is placeholder/too vague/non-informative for this question.
- Mark valid if it provides at least meaningful directional detail for this question.
- Keep ack neutral and accurate to the answer.
- Do not invent facts.
"""

    raw = st.session_state.generator.ask(prompt, history=[], use_rag=False)

    try:
        parsed = json.loads(raw)
        is_valid = bool(parsed.get("is_valid", False))
        ack = str(parsed.get("ack", "")).strip()
        feedback = str(parsed.get("feedback", "")).strip()

        if not ack:
            ack = "Thanks, noted."
        if not is_valid and not feedback:
            feedback = "Please provide a bit more specific detail for this question."

        return {
            "is_valid": is_valid,
            "ack": ack,
            "feedback": feedback,
        }
    except Exception:
        return {
            "is_valid": len(answer.strip()) >= 8,
            "ack": "Thanks, noted.",
            "feedback": "Please provide a bit more detail for this question.",
        }


def render_guided_progress_footer(current, total, progress_value):
    pct = int(max(0.0, min(progress_value, 1.0)) * 100)
    st.markdown(
        f"""
        <style>
        .guided-progress-footer {{
            position: fixed;
            right: 1rem;
            top: 6.5rem;
            width: min(340px, 42vw);
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 0.5rem 0.75rem;
            z-index: 9999;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        }}
        .guided-progress-label {{
            font-size: 0.85rem;
            margin-bottom: 0.35rem;
        }}
        .guided-progress-track {{
            width: 100%;
            height: 8px;
            background: #e5e7eb;
            border-radius: 999px;
            overflow: hidden;
        }}
        .guided-progress-fill {{
            height: 100%;
            width: {pct}%;
            background: #1f77d0;
        }}
        </style>
        <div class="guided-progress-footer">
            <div class="guided-progress-label">Guided interview progress: Question {current} of {total}</div>
            <div class="guided-progress-track">
                <div class="guided-progress-fill"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def start_guided_mode():
    st.session_state.mode = "guided"
    st.session_state.messages = []
    st.session_state.guided_step = 0
    st.session_state.guided_answers = []
    st.session_state.guided_questions = generate_guided_questions()
    st.session_state.generator.conversation_memory = {
        "last_query": None,
        "last_retrieved": None,
        "pending_concept_query": None,
        "active_faculty_name": None,
    }

    # LLM-generated introduction message (no RAG)
    intro_prompt = (
        "You are the BSU Graduate Advisor AI. Write a short welcome message for guided mode.\n"
        "Requirements:\n"
        "- Say you will ask 7 questions.\n"
        "- Say the goal is to recommend 1 to 3 faculty advisors.\n"
        "- Keep it short (3 to 6 sentences).\n"
        "- Do not ask any of the 7 questions yet.\n"
    )

    with st.spinner("Starting guided mode..."):
        intro_text = st.session_state.generator.ask(
            intro_prompt,
            history=[],
            use_rag=False,
        )

    st.session_state.messages.append({"role": "assistant", "content": intro_text})

    # Ask first guided question
    st.session_state.messages.append({"role": "assistant", "content": st.session_state.guided_questions[0]})
