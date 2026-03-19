# chatbot_demo.py
# Streamlit UI for BSU Graduate Advisor AI

import streamlit as st
from full_rag import ResponseEngine
import os


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

def reset_to_welcome():
    st.session_state.mode = "welcome"
    st.session_state.messages = []
    st.session_state.guided_step = 0
    st.session_state.guided_answers = []


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
    st.session_state.messages.append({"role": "assistant", "content": GUIDED_QUESTIONS[0]})


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
    st.markdown("### Hello. I’m your BSU advisor assistant for CS graduate students.")
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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

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

            st.session_state.mode = "explore"
            st.rerun()


# ==================== ROUTING ====================

if st.session_state.mode == "welcome":
    render_welcome_screen()
elif st.session_state.mode == "guided":
    render_guided()
else:
    render_chat()


# # chatbot_demo.py
# # This is the main file that creates the chatbot user interface
# # It uses Streamlit to create a web-based chat interface

# import streamlit as st
# from full_rag import ResponseEngine
# import os


# # ==================== PAGE CONFIGURATION ====================
# # This sets up how the webpage looks and behaves

# st.set_page_config(
#     page_title="BSU Advisor AI",  # Title shown in browser tab
#     layout="centered",  # Content is centered on page (not full width)
#     page_icon="assets/bsu_logo.png",  # Icon in browser tab
#     initial_sidebar_state="collapsed"  # Sidebar is collapsed when page loads
# )

# # ==================== CUSTOM STYLING ====================
# # This adds custom CSS to make the chat look better

# st.markdown(
#     """
#     <style>
#     /* Style for chat message bubbles */
#     .stChatMessage {
#         padding: 1rem;  /* Space inside messages */
#         border-radius: 0.5rem;  /* Rounded corners */
#     }
#     /* Style for main content area */
#     .main {
#         padding: 2rem;  /* Space around content */
#     }
#     </style>
# """,
#     unsafe_allow_html=True,
# )  # Allow HTML/CSS in markdown


# # ==================== HEADER (SAME AS BEFORE) ====================
# # Logo and title at the top of the page

# col1, col2 = st.columns([1, 7])

# with col1:
#     st.image("assets/bsu_logo.png", width=80)

# with col2:
#     st.title("BSU Graduate Advisor AI")
#     st.caption("Your intelligent assistant for BSU CS graduate advising")


# # ==================== API TOKEN CHECK ====================
# # Your baseline uses BSU_API_KEY (not GROQ_API_KEY)

# if not os.getenv("BSU_API_KEY"):
#     st.error("BSU_API_KEY not found!")
#     st.info("Please create a .env file with your BSU API key inside it.")
#     st.stop()


# # ==================== SESSION STATE INITIALIZATION ====================

# if "mode" not in st.session_state:
#     # "welcome" | "explore" | "guided"
#     st.session_state.mode = "welcome"

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if "generator" not in st.session_state:
#     with st.spinner("Initializing AI assistant..."):
#         st.session_state.generator = ResponseEngine()

# # Guided mode state
# if "guided_step" not in st.session_state:
#     st.session_state.guided_step = 0  # 0..6
# if "guided_answers" not in st.session_state:
#     st.session_state.guided_answers = []  # list of 7 strings


# GUIDED_QUESTIONS = [
#     "What are your top 2 to 3 research interests right now? (Examples: NLP, ML, HCI, security, systems, CV)",
#     "What graduate or senior-level courses have you taken that you liked? List a few.",
#     "What was your bachelor’s degree and what kind of projects did you do in undergrad?",
#     "What are your strongest skills or tools? (Languages, frameworks, math, writing, systems, ML, etc.)",
#     "What topics or project types do you NOT want?",
#     "What is your goal for grad school? (Thesis vs project, research vs industry, PhD interest, publish, build product)",
#     "Any constraints or preferences? (Funding need, time, applied vs theoretical, hands-on vs independent advising)",
# ]


# def reset_to_welcome():
#     st.session_state.mode = "welcome"
#     st.session_state.messages = []
#     st.session_state.guided_step = 0
#     st.session_state.guided_answers = []


# def start_guided_mode():
#     st.session_state.mode = "guided"
#     st.session_state.messages = []
#     st.session_state.guided_step = 0
#     st.session_state.guided_answers = []
#     # Ask first guided question as assistant message
#     st.session_state.messages.append({"role": "assistant", "content": GUIDED_QUESTIONS[0]})


# def build_student_profile_text(answers):
#     # answers is a list of 7 strings in order
#     # Structured, label-based profile is best for embedding + cosine retrieval
#     profile = []
#     profile.append("STUDENT PROFILE (from guided interview)")
#     profile.append("")
#     profile.append(f"1) Research interests: {answers[0].strip()}")
#     profile.append(f"2) Courses liked: {answers[1].strip()}")
#     profile.append(f"3) Bachelor’s degree and undergrad projects: {answers[2].strip()}")
#     profile.append(f"4) Strongest skills/tools: {answers[3].strip()}")
#     profile.append(f"5) Not interested in: {answers[4].strip()}")
#     profile.append(f"6) Grad school goal: {answers[5].strip()}")
#     profile.append(f"7) Constraints/preferences: {answers[6].strip()}")
#     profile.append("")
#     profile.append(
#         "TASK: Based on the student profile above, recommend 1 to 3 BSU CS faculty advisors. "
#         "Ground each recommendation in the provided faculty evidence (chunks)."
#     )
#     return "\n".join(profile)


# # ==================== SIDEBAR (SAME AS BEFORE, ONLY IN CHAT MODES) ====================


# def render_sidebar():
#     with st.sidebar:
#         st.header("Not sure where to start?")
#         st.markdown(
#             """
#             Not sure how to choose a graduate advisor? You are not alone.

#             This assistant helps you explore faculty, research areas, and advising options.
#             There is no right way to start, just ask what you are curious about.
#             """
#         )
#         st.markdown(
#             """

#         **Try asking:**
#         - "Hello, how are you?"
#         - "List all faculty with their research areas."
#         - "What faculty does AI research?"
#         - "Tell me about Dr. Xinyi Zhou"
#         - "How do I choose an advisor?"
#         """
#         )

#         st.divider()

#         if st.button("Start over"):
#             reset_to_welcome()
#             st.rerun()


# # ==================== WELCOME SCREEN ====================


# def render_welcome_screen():
#     st.markdown("### Hello, I’m your BSU advisor for CS graduate students.")
#     st.markdown(
#         "I can help you explore BSU CS faculty, understand research areas, and figure out who might be a good advisor match."
#     )
#     st.markdown("Choose how you want to start:")

#     b1, b2 = st.columns(2)

#     with b1:
#         if st.button("Start from scratch", use_container_width=True):
#             start_guided_mode()
#             st.rerun()

#     with b2:
#         if st.button("I wanna ask questions", use_container_width=True):
#             st.session_state.mode = "explore"
#             st.session_state.messages = []
#             # Same welcome message you had before
#             st.session_state.messages.append(
#                 {
#                     "role": "assistant",
#                     "content": "Hi! I'm your BSU Graduate Advisor AI. I can help you learn about CS faculty, their research areas, availability, and guide you through the advisor selection process. What would you like to know?",
#                 }
#             )
#             st.rerun()


# # ==================== CHAT UI (UNCHANGED LOOK) ====================


# def render_chat():
#     render_sidebar()

#     # Display chat history
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.write(message["content"])

#     # Chat input
#     if user_query := st.chat_input("Ask me anything about graduate advising at BSU."):
#         # Display user message
#         with st.chat_message("user"):
#             st.write(user_query)

#         # Add user message to history
#         st.session_state.messages.append({"role": "user", "content": user_query})

#         # Generate AI response
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking."):
#                 answer = st.session_state.generator.ask(
#                     user_query,
#                     history=st.session_state.messages[:-1],
#                     use_rag=True,
#                 )
#             st.write(answer)

#         # Add AI response to history
#         st.session_state.messages.append({"role": "assistant", "content": answer})

#     # Footer (same style as your file)
#     st.markdown("---")
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.caption("BSU CS Advisor Assistant")
#     with col2:
#         st.caption("Baseline RAG + Chunk Retrieval")
#     with col3:
#         st.caption("BoiseState.ai API")


# # ==================== GUIDED MODE UI ====================


# def render_guided():
#     render_sidebar()

#     # Display chat history
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.write(message["content"])

#     # Chat input (answers to guided questions)
#     if user_answer := st.chat_input("Answer the guided question to continue."):
#         # Display user message
#         with st.chat_message("user"):
#             st.write(user_answer)

#         # Add user answer to history
#         st.session_state.messages.append({"role": "user", "content": user_answer})

#         # Store answer and advance
#         st.session_state.guided_answers.append(user_answer)
#         st.session_state.guided_step += 1

#         # If there are more questions, ask the next one
#         if st.session_state.guided_step < len(GUIDED_QUESTIONS):
#             next_q = GUIDED_QUESTIONS[st.session_state.guided_step]
#             st.session_state.messages.append({"role": "assistant", "content": next_q})
#             st.rerun()

#         # Otherwise, build profile and generate recommendations, then switch to explore mode
#         if st.session_state.guided_step == len(GUIDED_QUESTIONS):
#             profile_text = build_student_profile_text(st.session_state.guided_answers)

#             with st.chat_message("assistant"):
#                 with st.spinner("Building your profile and finding matches."):
#                     answer = st.session_state.generator.ask(
#                         profile_text,
#                         history=st.session_state.messages,
#                         use_rag=True,
#                     )
#                 st.write(answer)

#             st.session_state.messages.append({"role": "assistant", "content": answer})

#             # Switch back to normal ask-questions mode, keeping history
#             st.session_state.mode = "explore"
#             st.rerun()


# # ==================== ROUTING ====================

# if st.session_state.mode == "welcome":
#     render_welcome_screen()
# elif st.session_state.mode == "guided":
#     render_guided()
# else:
#     render_chat()
    
#     # # chatbot_demo.py
# # # This is the main file that creates the chatbot user interface
# # # It uses Streamlit to create a web-based chat interface

# # import streamlit as st
# # from full_rag import ResponseEngine
# # import os


# # # ==================== PAGE CONFIGURATION ====================
# # # This sets up how the webpage looks and behaves

# # st.set_page_config(
# #     page_title="BSU Advisor AI",  # Title shown in browser tab
# #     layout="centered",  # Content is centered on page (not full width)
# #     page_icon="assets/bsu_logo.png",  # Icon in browser tab
# #     initial_sidebar_state="collapsed"  # Sidebar is collapsed when page loads
# # )

# # # ==================== CUSTOM STYLING ====================
# # # This adds custom CSS to make the chat look better

# # st.markdown("""
# #     <style>
# #     /* Style for chat message bubbles */
# #     .stChatMessage {
# #         padding: 1rem;  /* Space inside messages */
# #         border-radius: 0.5rem;  /* Rounded corners */
# #     }
# #     /* Style for main content area */
# #     .main {
# #         padding: 2rem;  /* Space around content */
# #     }
# #     </style>
# # """, unsafe_allow_html=True)  # Allow HTML/CSS in markdown


# # # ==================== HEADER (SAME AS BEFORE) ====================
# # # Logo and title at the top of the page

# # col1, col2 = st.columns([1, 7])

# # with col1:
# #     st.image("assets/bsu_logo.png", width=80)

# # with col2:
# #     st.title("BSU Graduate Advisor AI")
# #     st.caption("Your intelligent assistant for BSU CS graduate advising")


# # # ==================== API TOKEN CHECK ====================
# # # Your baseline uses BSU_API_KEY (not GROQ_API_KEY)

# # if not os.getenv("BSU_API_KEY"):
# #     st.error("BSU_API_KEY not found!")
# #     st.info("Please create a .env file with your BSU API key inside it.")
# #     st.stop()


# # # ==================== SESSION STATE INITIALIZATION ====================

# # if "mode" not in st.session_state:
# #     # "welcome" | "explore" | "guided"
# #     st.session_state.mode = "welcome"

# # if "messages" not in st.session_state:
# #     st.session_state.messages = []

# # if "generator" not in st.session_state:
# #     with st.spinner("Initializing AI assistant..."):
# #         st.session_state.generator = ResponseEngine()


# # def reset_to_welcome():
# #     st.session_state.mode = "welcome"
# #     st.session_state.messages = []


# # # ==================== SIDEBAR (SAME AS BEFORE, ONLY IN CHAT MODES) ====================

# # def render_sidebar():
# #     with st.sidebar:
# #         st.header("Not sure where to start?")
# #         st.markdown("""
# #             Not sure how to choose a graduate advisor? You are not alone.

# #             This assistant helps you explore faculty, research areas, and advising options.
# #             There is no right way to start, just ask what you are curious about.
# #             """)
# #         st.markdown("""

# #         **Try asking:**
# #         - "Hello, how are you?"
# #         - "List all faculty with their research areas."
# #         - "What faculty does AI research?"
# #         - "Tell me about Dr. Xinyi Zhou"
# #         - "How do I choose an advisor?"
# #         """)

# #         st.divider()

# #         if st.button("Start over"):
# #             reset_to_welcome()
# #             st.rerun()


# # # ==================== WELCOME SCREEN ====================

# # def render_welcome_screen():
# #     st.markdown("### Hello, I’m your BSU advisor for CS graduate students.")
# #     st.markdown(
# #         "I can help you explore BSU CS faculty, understand research areas, and figure out who might be a good advisor match."
# #     )
# #     st.markdown("Choose how you want to start:")

# #     b1, b2 = st.columns(2)

# #     with b1:
# #         if st.button("Start from scratch", use_container_width=True):
# #             # We will implement guided mode next. For now, route to guided mode state.
# #             st.session_state.mode = "guided"
# #             st.session_state.messages = []
# #             # Optional: initial assistant message
# #             st.session_state.messages.append({
# #                 "role": "assistant",
# #                 "content": "Guided discovery mode is coming next. For now, ask me anything, or click “I wanna ask questions”."
# #             })
# #             st.rerun()

# #     with b2:
# #         if st.button("I wanna ask questions", use_container_width=True):
# #             st.session_state.mode = "explore"
# #             st.session_state.messages = []
# #             # Same welcome message you had before
# #             st.session_state.messages.append({
# #                 "role": "assistant",
# #                 "content": "Hi! I'm your BSU Graduate Advisor AI. I can help you learn about CS faculty, their research areas, availability, and guide you through the advisor selection process. What would you like to know?"
# #             })
# #             st.rerun()


# # # ==================== CHAT UI (UNCHANGED LOOK) ====================

# # def render_chat():
# #     render_sidebar()

# #     # Display chat history
# #     for message in st.session_state.messages:
# #         with st.chat_message(message["role"]):
# #             st.write(message["content"])

# #     # Chat input
# #     if user_query := st.chat_input("Ask me anything about graduate advising at BSU."):
# #         # Display user message
# #         with st.chat_message("user"):
# #             st.write(user_query)

# #         # Add user message to history
# #         st.session_state.messages.append({
# #             "role": "user",
# #             "content": user_query
# #         })

# #         # Generate AI response
# #         with st.chat_message("assistant"):
# #             with st.spinner("Thinking."):
# #                 answer = st.session_state.generator.ask(
# #                     user_query,
# #                     history=st.session_state.messages[:-1],
# #                     use_rag=True
# #                 )
# #             st.write(answer)

# #         # Add AI response to history
# #         st.session_state.messages.append({
# #             "role": "assistant",
# #             "content": answer
# #         })

# #     # Footer (same style as your file)
# #     st.markdown("---")
# #     col1, col2, col3 = st.columns(3)
# #     with col1:
# #         st.caption("BSU CS Advisor Assistant")
# #     with col2:
# #         st.caption("Baseline RAG + Chunk Retrieval")
# #     with col3:
# #         st.caption("BoiseState.ai API")


# # # ==================== ROUTING ====================

# # if st.session_state.mode == "welcome":
# #     render_welcome_screen()
# # else:
# #     # For now, guided uses the same chat UI until we implement it
# #     render_chat()