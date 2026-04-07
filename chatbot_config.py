import streamlit as st
from rag_engine import ResponseEngine
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
