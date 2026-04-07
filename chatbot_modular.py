from chatbot_config import st
from chatbot_views import render_welcome_screen, render_guided, render_chat


# ==================== ROUTING ====================

if st.session_state.mode == "welcome":
    render_welcome_screen()
elif st.session_state.mode == "guided":
    render_guided()
else:
    render_chat()
