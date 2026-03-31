# chatbot_demo.py
# Streamlit UI for BSU Graduate Advisor AI

import streamlit as st
from full_rag_fixed import ResponseEngine
import os
import json


# ==================== PAGE CONFIGURATION ====================

st.set_page_config(
    page_title="BSU Advisor AI",
    layout="centered",
    page_icon="assets/bsu_logo.png",
    initial_sidebar_state="collapsed",
)

# ==================== CUSTOM STYLING ====================

st.markdown(
    """
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .main {
        padding: 2rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ==================== HEADER ====================

col1, col2 = st.columns([1, 7])

with col1:
    st.image("assets/bsu_logo.png", width=80)

with col2:
    st.title("BSU Graduate Advisor AI")
    st.caption("Your assistant for BSU CS graduate advising")


# ==================== API TOKEN CHECK ====================

if not os.getenv("BSU_API_KEY"):
    st.error("BSU_API_KEY not found!")
    st.info("Please create a .env file with your BSU API key inside it.")
    st.stop()


# ==================== SESSION STATE INITIALIZATION ====================

if "mode" not in st.session_state:
    # "welcome" | "explore" | "guided"
    st.session_state.mode = "welcome"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "generator" not in st.session_state:
    with st.spinner("Initializing AI assistant..."):
        st.session_state.generator = ResponseEngine()

# Guided mode state
if "guided_step" not in st.session_state:
    st.session_state.guided_step = 0  # 0..6
if "guided_answers" not in st.session_state:
    st.session_state.guided_answers = []  # list of 7 strings
if "guided_questions" not in st.session_state:
    st.session_state.guided_questions = []


# ==================== GUIDED QUESTIONS (FINAL) ====================

GUIDED_QUESTIONS = [
    "What is your CS graduate program or emphasis? (Examples: MS in CS, AI emphasis, Cybersecurity emphasis, Systems emphasis)",
    "What are your top 2 to 3 research interests right now? (Examples: NLP, ML, HCI, security, systems, CV)",
    "What graduate or senior-level courses have you taken that you liked? List a few.",
    "What was your bachelor's degree and what kind of projects did you do in undergrad?",
    "What are your strongest skills or tools? (Languages, frameworks, math, writing, systems, ML, etc.)",
    "What topics or project types do you NOT want?",
    "What is your goal for grad school? (Thesis vs project, research vs industry, PhD interest, publish, build product)",
]


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


# ==================== SIDEBAR ====================

def render_sidebar():
    with st.sidebar:
        st.header("Not sure where to start?")
        st.markdown(
            """
Not sure how to choose a graduate advisor? You are not alone.

This assistant helps you explore faculty and research areas, and narrow down advisor matches.
            """
        )

        st.markdown(
            """
**Try asking:**
- "List all faculty with their research areas."
- "Who works on computer vision?"
- "Tell me about Dr. X"
- "What faculty do NLP?"
            """
        )

        st.divider()

        if st.button("Start over"):
            reset_to_welcome()
            st.rerun()


# ==================== WELCOME SCREEN ====================

def render_welcome_screen():
    st.markdown("### Hello. I'm your BSU advisor assistant for CS graduate students.")
    st.markdown(
        "Choose how you want to start. Guided mode is a short interview. Ask questions mode is free-form chat."
    )

    b1, b2 = st.columns(2)

    with b1:
        if st.button("Start from scratch", use_container_width=True):
            start_guided_mode()
            st.rerun()

        st.caption(
            "Guided mode: answer 7 questions and get 1 to 3 advisor recommendations based on your profile."
        )

    with b2:
        if st.button("I wanna ask questions", use_container_width=True):
            st.session_state.mode = "explore"
            st.session_state.messages = []
            st.session_state.generator.conversation_memory = {
                "last_query": None,
                "last_retrieved": None,
                "pending_concept_query": None,
                "active_faculty_name": None,
            }
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "Hi! I'm your BSU Graduate Advisor AI. I can help you learn about CS faculty and their research areas. "
                        "What would you like to know?"
                    ),
                }
            )
            st.rerun()

        st.caption(
            "Ask questions mode: jump straight into questions about faculty, topics, and advisor matching."
        )


# ==================== EXPLORE CHAT UI ====================

def render_chat():
    render_sidebar()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if user_query := st.chat_input("Ask me anything about graduate advising at BSU."):
        with st.chat_message("user"):
            st.write(user_query)

        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Thinking."):
                answer = st.session_state.generator.ask(
                    user_query,
                    history=st.session_state.messages[:-1],
                    use_rag=True,
                )
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    st.markdown("---")


# ==================== GUIDED MODE UI ====================

def render_guided():
    render_sidebar()

    questions = st.session_state.guided_questions or GUIDED_QUESTIONS
    current = min(st.session_state.guided_step + 1, len(questions))
    total = len(questions)
    progress_value = st.session_state.guided_step / total

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if user_answer := st.chat_input("Answer the guided question to continue."):
        user_answer = user_answer.strip()
        if not user_answer:
            st.warning("Please enter an answer before continuing.")
            return

        current_question = questions[st.session_state.guided_step] if st.session_state.guided_step < len(questions) else ""
        assessment = assess_guided_answer(current_question, user_answer)
        if not assessment["is_valid"]:
            st.warning(assessment["feedback"])
            return

        with st.chat_message("user"):
            st.write(user_answer)

        st.session_state.messages.append({"role": "user", "content": user_answer})
        st.session_state.messages.append({"role": "assistant", "content": assessment["ack"]})

        st.session_state.guided_answers.append(user_answer)
        st.session_state.guided_step += 1

        if st.session_state.guided_step < len(questions):
            next_q = questions[st.session_state.guided_step]
            st.session_state.messages.append({"role": "assistant", "content": next_q})
            st.rerun()

        if st.session_state.guided_step == len(questions):
            profile_text = build_student_profile_text(st.session_state.guided_answers)

            with st.chat_message("assistant"):
                with st.spinner("Summarizing and finding advisor matches..."):
                    answer = st.session_state.generator.ask(
                        profile_text,
                        history=st.session_state.messages,
                        use_rag=True,
                    )
                st.write(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I can give you more information about a faculty in particular or answer other questions you might have. What would you like to do next?"
            })
            st.session_state.mode = "explore"
            st.rerun()

    render_guided_progress_footer(current, total, progress_value)


# ==================== ROUTING ====================

if st.session_state.mode == "welcome":
    render_welcome_screen()
elif st.session_state.mode == "guided":
    render_guided()
else:
    render_chat()