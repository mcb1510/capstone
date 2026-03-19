# chatbot_demo.py
import os
import re
import streamlit as st
from full_rag import ResponseEngine

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

# ==================== SESSION STATE INIT ====================

if "mode" not in st.session_state:
    st.session_state.mode = "welcome"  # "welcome" | "guided" | "explore"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "generator" not in st.session_state:
    with st.spinner("Initializing AI assistant..."):
        st.session_state.generator = ResponseEngine()

# Guided state
if "guided_step" not in st.session_state:
    st.session_state.guided_step = 0
if "guided_answers" not in st.session_state:
    st.session_state.guided_answers = []

# Explore follow-up controller state
if "pending_followup" not in st.session_state:
    st.session_state.pending_followup = None  # dict or None
if "last_followup_decision" not in st.session_state:
    st.session_state.last_followup_decision = ""
if "last_routed_query" not in st.session_state:
    st.session_state.last_routed_query = ""

# Debug toggle
if "debug_followups" not in st.session_state:
    st.session_state.debug_followups = True

# ==================== GUIDED QUESTIONS (FINAL) ====================

GUIDED_QUESTIONS = [
    "What is your CS graduate program or emphasis? (Examples: MS in CS, AI emphasis, Cybersecurity emphasis, Systems emphasis)",
    "What are your top 2 to 3 research interests right now? (Examples: NLP, ML, HCI, security, systems, CV)",
    "What graduate or senior-level courses have you taken that you liked? List a few.",
    "What was your bachelor’s degree and what kind of projects did you do in undergrad?",
    "What are your strongest skills or tools? (Languages, frameworks, math, writing, systems, ML, etc.)",
    "What topics or project types do you NOT want?",
    "What is your goal for grad school? (Thesis vs project, research vs industry, PhD interest, publish, build product)",
]

# ==================== HELPERS ====================


YES_SET = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "help me", "yes help me", "yes, help me"}
NO_SET = {"no", "n", "nope", "nah"}


def reset_to_welcome():
    st.session_state.mode = "welcome"
    st.session_state.messages = []
    st.session_state.guided_step = 0
    st.session_state.guided_answers = []
    st.session_state.pending_followup = None
    st.session_state.last_followup_decision = ""
    st.session_state.last_routed_query = ""


def strip_followup_tag(text: str) -> str:
    # Removes any hidden followup tag if present
    return re.sub(r"\[FOLLOWUP[^\]]*\]\s*$", "", text or "").rstrip()


def parse_followup_tag(text: str):
    """
    Optional hidden tag format:
      [FOLLOWUP type=faculty_recs topic=robotics entity=Tim_Andersen]
    Values can be omitted.
    """
    if not text:
        return None
    m = re.search(r"\[FOLLOWUP\s+([^\]]+)\]\s*$", text.strip())
    if not m:
        return None
    payload = m.group(1).strip()
    meta = {}
    for part in payload.split():
        if "=" in part:
            k, v = part.split("=", 1)
            meta[k.strip()] = v.strip()
    if not meta.get("type"):
        return None
    # decode underscores
    if "entity" in meta:
        meta["entity"] = meta["entity"].replace("_", " ")
    if "topic" in meta:
        meta["topic"] = meta["topic"].replace("_", " ")
    return meta


def normalize_interest_text(s: str) -> str:
    """
    Converts "I like AI" -> "AI", "I'm interested in deep learning" -> "deep learning"
    Very small normalization, not aggressive.
    """
    if not s:
        return ""
    t = s.strip()
    t_low = t.lower()
    prefixes = [
        "i like ",
        "i love ",
        "im interested in ",
        "i'm interested in ",
        "i am interested in ",
        "i want ",
        "im into ",
        "i'm into ",
        "interested in ",
    ]
    for p in prefixes:
        if t_low.startswith(p):
            return t[len(p):].strip()
    return t


def get_last_user_nontrivial(history):
    for m in reversed(history):
        if m.get("role") != "user":
            continue
        txt = (m.get("content") or "").strip()
        if not txt:
            continue
        low = txt.lower().strip()
        if low in YES_SET or low in NO_SET:
            continue
        return txt
    return ""


def detect_followup_meta_from_assistant(assistant_text: str, history):
    """
    Sets pending followup based on assistant response content.
    Uses explicit [FOLLOWUP ...] tag if present, else uses heuristics.
    """
    raw = assistant_text or ""
    tagged = parse_followup_tag(raw)
    if tagged:
        return {
            "type": tagged.get("type"),
            "topic": tagged.get("topic"),
            "entity": tagged.get("entity"),
            "asked_text": strip_followup_tag(raw),
        }

    text = raw.lower()

    # Outreach question offer
    if ("prepare questions" in text or "reach out" in text) and ("dr." in text or "dr " in text):
        # try to extract "Dr. First Last"
        m = re.search(r"dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", raw)
        entity = m.group(1).strip() if m else None
        if entity:
            return {"type": "outreach", "entity": entity, "topic": None, "asked_text": raw}

    # Faculty recommendation offer
    if "recommend" in text and "faculty" in text:
        last_user = get_last_user_nontrivial(history)
        topic = normalize_interest_text(last_user)
        if topic:
            return {"type": "faculty_recs", "entity": None, "topic": topic, "asked_text": raw}
        return {"type": "faculty_recs", "entity": None, "topic": None, "asked_text": raw}

    # Topic refinement offer, common phrasing
    if "specific area" in text or "which area" in text or "what specific type" in text:
        last_user = get_last_user_nontrivial(history)
        topic = normalize_interest_text(last_user)
        if topic:
            return {"type": "topic_refine", "entity": None, "topic": topic, "asked_text": raw}
        return {"type": "topic_refine", "entity": None, "topic": None, "asked_text": raw}

    return None


def is_clear_interrupt(user_text: str) -> bool:
    """
    Detect obvious topic switches. This is intentionally conservative.
    If it returns True, we treat it as a new request.
    """
    if not user_text:
        return False
    t = user_text.strip().lower()

    # User explicitly signals a new topic
    explicit = ["new question", "different topic", "separate question", "unrelated", "by the way"]
    if any(t.startswith(x) for x in explicit):
        return True

    # Some clearly different-life domains (expand if needed)
    unrelated_keywords = [
        "cpt", "opt", "visa", "i-20", "i-94", "tax", "1098", "resume", "cover letter",
        "salary", "job offer", "interview", "rent", "apartment", "car", "gym",
    ]
    if any(k in t for k in unrelated_keywords):
        return True

    return False


def route_explore_message(user_text: str, history):
    """
    Decide whether user_text is answering a pending follow-up or starting a new request.
    Returns (routed_query, used_pending: bool)
    """
    pending = st.session_state.pending_followup
    st.session_state.last_followup_decision = ""
    st.session_state.last_routed_query = ""

    if not pending:
        st.session_state.last_followup_decision = "no pending follow-up"
        st.session_state.last_routed_query = user_text
        return user_text, False

    # If user clearly interrupts, drop pending
    if is_clear_interrupt(user_text):
        st.session_state.pending_followup = None
        st.session_state.last_followup_decision = "interrupt (clear topic switch)"
        st.session_state.last_routed_query = user_text
        return user_text, False

    u = (user_text or "").strip()
    u_low = u.lower()

    # If the reply is a short confirmation, treat as continuing the follow-up
    if u_low in YES_SET or u_low in NO_SET or len(u_low.split()) <= 3:
        # Continue unless it is a clear new question with '?'
        # Short questions like "who?" still should likely continue.
        decision = "continue (short reply)"
        follow_type = pending.get("type")

        if follow_type == "outreach":
            entity = pending.get("entity") or "the professor"
            routed = (
                f"Draft 6 to 8 concise outreach questions I can email to Dr. {entity} "
                f"to ask about advising and project/capstone fit. Keep them practical and professional."
            )
            st.session_state.last_followup_decision = decision + " -> outreach"
            st.session_state.last_routed_query = routed
            st.session_state.pending_followup = None
            return routed, True

        if follow_type in {"faculty_recs", "topic_refine"}:
            topic = pending.get("topic") or normalize_interest_text(get_last_user_nontrivial(history))
            if not topic:
                topic = "the topic we were just discussing"
            # If user said "no", treat as interrupt and ask what they want next
            if u_low in NO_SET:
                routed = "Okay. What topic should I match BSU CS faculty to?"
                st.session_state.last_followup_decision = "continue (short reply) -> user declined"
                st.session_state.last_routed_query = routed
                st.session_state.pending_followup = {"type": "topic_refine", "topic": None, "entity": None, "asked_text": ""}
                return routed, True

            # Default: recommend faculty for topic
            routed = f"BSU faculty that work with {topic}"
            st.session_state.last_followup_decision = decision + " -> faculty recs"
            st.session_state.last_routed_query = routed
            st.session_state.pending_followup = None
            return routed, True

        # Fallback: treat as new request
        st.session_state.last_followup_decision = "interrupt (unknown follow-up type)"
        st.session_state.last_routed_query = user_text
        st.session_state.pending_followup = None
        return user_text, False

    # For longer replies, decide by overlap with pending topic/entity
    follow_type = pending.get("type")
    topic = (pending.get("topic") or "").lower()
    entity = (pending.get("entity") or "").lower()
    u_low = u_low

    overlap = False
    if topic and topic in u_low:
        overlap = True
    if entity and entity in u_low:
        overlap = True

    if overlap:
        # Continue follow-up and enrich topic with the new detail
        if follow_type in {"faculty_recs", "topic_refine"}:
            base = pending.get("topic") or ""
            routed = f"Recommend BSU CS faculty based on: {base}. Details: {u}"
            st.session_state.last_followup_decision = "continue (topic/entity overlap)"
            st.session_state.last_routed_query = routed
            st.session_state.pending_followup = None
            return routed, True

        if follow_type == "outreach":
            ent = pending.get("entity") or "the professor"
            routed = (
                f"Draft 6 to 8 concise outreach questions I can email to Dr. {ent}. "
                f"User preferences for the questions: {u}"
            )
            st.session_state.last_followup_decision = "continue (topic/entity overlap) -> outreach"
            st.session_state.last_routed_query = routed
            st.session_state.pending_followup = None
            return routed, True

    # If no overlap, treat as interrupt (new request)
    st.session_state.last_followup_decision = "interrupt (no overlap with follow-up)"
    st.session_state.last_routed_query = user_text
    st.session_state.pending_followup = None
    return user_text, False


def build_guided_profile_query(answers):
    # Force the classifier into the recommendation path by making intent explicit up front.
    return f"""Recommend 1 to 3 BSU CS faculty advisors based on the student profile below.
First, write a short paragraph summarizing the student.
Then recommend 1 to 3 advisors and explain why, grounded in the retrieved faculty evidence (chunks) only.

STUDENT PROFILE
Program/emphasis: {answers[0].strip()}
Top interests: {answers[1].strip()}
Courses liked: {answers[2].strip()}
Bachelor’s + undergrad projects: {answers[3].strip()}
Strongest skills/tools: {answers[4].strip()}
Do NOT want: {answers[5].strip()}
Goal for grad school: {answers[6].strip()}
"""


def start_guided_mode():
    st.session_state.mode = "guided"
    st.session_state.messages = []
    st.session_state.guided_step = 0
    st.session_state.guided_answers = []
    st.session_state.pending_followup = None
    st.session_state.last_followup_decision = ""
    st.session_state.last_routed_query = ""

    intro_prompt = (
        "You are the BSU Graduate Advisor AI.\n"
        "Write a short welcome message for guided mode.\n"
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
    st.session_state.messages.append({"role": "assistant", "content": GUIDED_QUESTIONS[0]})


def render_sidebar():
    with st.sidebar:
        st.header("Help")

        st.session_state.debug_followups = st.toggle(
            "Show debug",
            value=st.session_state.debug_followups,
        )

        if st.session_state.debug_followups:
            st.subheader("Follow-up state")
            st.write("Mode:", st.session_state.mode)
            st.write("Pending:", st.session_state.pending_followup)
            st.write("Last decision:", st.session_state.last_followup_decision)
            st.write("Last routed query:", st.session_state.last_routed_query)

        st.divider()

        st.markdown(
            """
Try asking:
- "List all faculty with their research areas."
- "Who works on computer vision?"
- "Tell me about Dr. X"
- "BSU faculty that work with deep learning"
            """
        )

        st.divider()

        if st.button("Start over"):
            reset_to_welcome()
            st.rerun()


# ==================== WELCOME SCREEN ====================

def render_welcome_screen():
    st.markdown("### Hello. I’m your BSU advisor assistant for CS graduate students.")
    st.markdown("Choose how you want to start:")

    b1, b2 = st.columns(2)

    with b1:
        if st.button("Start from scratch", use_container_width=True):
            start_guided_mode()
            st.rerun()

        st.caption(
            "Guided mode runs a short fixed interview (7 questions), then recommends 1 to 3 advisors."
        )

    with b2:
        if st.button("I wanna ask questions", use_container_width=True):
            st.session_state.mode = "explore"
            st.session_state.messages = []
            st.session_state.pending_followup = None
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
            "Ask questions mode is free-form chat. You can switch topics anytime."
        )


# ==================== CHAT RENDER ====================

def render_messages():
    for m in st.session_state.messages:
        role = m.get("role")
        content = strip_followup_tag(m.get("content", ""))
        with st.chat_message(role):
            st.write(content)


# ==================== EXPLORE MODE ====================

def render_explore():
    render_sidebar()
    render_messages()

    if user_query := st.chat_input("Ask me anything about graduate advising at BSU."):
        # Route using follow-up controller (explore mode interruptible)
        routed_query, used_pending = route_explore_message(user_query, st.session_state.messages)

        # Add user message (what user typed)
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Call engine with the routed query
        with st.chat_message("assistant"):
            with st.spinner("Thinking."):
                answer = st.session_state.generator.ask(
                    routed_query,
                    history=st.session_state.messages[:-1],
                    use_rag=True,
                )
            st.write(strip_followup_tag(answer))

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Update pending follow-up state from assistant response
        meta = detect_followup_meta_from_assistant(answer, st.session_state.messages)
        st.session_state.pending_followup = meta

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("BSU CS Advisor Assistant")
    with c2:
        st.caption("Baseline RAG + Chunk Retrieval")
    with c3:
        st.caption("BoiseState.ai API")


# ==================== GUIDED MODE (NOT INTERRUPTIBLE) ====================

def render_guided():
    render_sidebar()
    render_messages()

    if user_answer := st.chat_input("Answer the guided question to continue."):
        with st.chat_message("user"):
            st.write(user_answer)

        st.session_state.messages.append({"role": "user", "content": user_answer})

        st.session_state.guided_answers.append(user_answer)
        st.session_state.guided_step += 1

        if st.session_state.guided_step < len(GUIDED_QUESTIONS):
            next_q = GUIDED_QUESTIONS[st.session_state.guided_step]
            st.session_state.messages.append({"role": "assistant", "content": next_q})
            st.rerun()

        if st.session_state.guided_step == len(GUIDED_QUESTIONS):
            profile_query = build_guided_profile_query(st.session_state.guided_answers)

            with st.chat_message("assistant"):
                with st.spinner("Summarizing and finding advisor matches..."):
                    answer = st.session_state.generator.ask(
                        profile_query,
                        history=st.session_state.messages,
                        use_rag=True,
                    )
                st.write(strip_followup_tag(answer))

            st.session_state.messages.append({"role": "assistant", "content": answer})

            # After guided finishes, go back to explore mode and keep history
            st.session_state.mode = "explore"
            # reset follow-up state after guided recommendation
            st.session_state.pending_followup = detect_followup_meta_from_assistant(answer, st.session_state.messages)
            st.rerun()


# ==================== ROUTING ====================

if st.session_state.mode == "welcome":
    render_welcome_screen()
elif st.session_state.mode == "guided":
    render_guided()
else:
    render_explore()