from chatbot_config import st, GUIDED_QUESTIONS
from chatbot_helpers import (
    reset_to_welcome,
    build_student_profile_text,
    assess_guided_answer,
    render_guided_progress_footer,
    start_guided_mode,
)


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
