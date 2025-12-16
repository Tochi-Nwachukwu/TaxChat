import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from src.retrieval_system import create_rag_chatbot
from src.embeddings_manager import load_vectorstore
from constants import db_name, MODEL
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="TaxChat",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Clean, minimal CSS
st.markdown("""
<style>
    /* Hide sidebar */
    [data-testid="stSidebar"], 
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }
    
    /* Hide Streamlit chrome */
    #MainMenu, footer, header {
        visibility: hidden;
    }
    
    /* Center content */
    .block-container {
        max-width: 800px;
        padding-top: 2rem;
    }
    
    /* Header */
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .header h1 {
        font-size: 2.2rem;
        color: white;
        margin-bottom: 0.3rem;
    }
    
    .header p {
        color: #666;
        font-size: 1rem;
    }
    
    /* Chat messages */
    .user-msg {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 12px 12px 4px 12px;
        margin: 0.8rem 0;
        border-left: 3px solid #2196f3;
    }
    
    .bot-msg {
        background: #f5f5f5;
        padding: 1rem;
        border-radius: 12px 12px 12px 4px;
        margin: 0.8rem 0;
        border-left: 3px solid #4caf50;
    }
    
    .msg-label {
        font-weight: 600;
        font-size: 0.85rem;
        color: #555;
        margin-bottom: 0.4rem;
    }
    
    .msg-content {
        color: #333;
        line-height: 1.5;
    }
    
    /* Source box */
    .source-item {
        background: black;
        border: 1px solid #ffcc02;
        border-radius: 8px;
        padding: 0.7rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    /* Quick questions */
    .quick-q {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: center;
        margin: 1rem 0;
    }
    
    /* Toolbar */
    .toolbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid #eee;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def init_chatbot():
    """Initialize and cache the chatbot - runs once on startup."""
    try:
        vectorstore = load_vectorstore(db_name=db_name)
        chatbot = create_rag_chatbot(vectorstore, model_name=MODEL, temperature=0.7)
        return chatbot, None
    except FileNotFoundError:
        return None, "Knowledge base not found. Please run `python main.py` first."
    except Exception as e:
        return None, str(e)


# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_sources" not in st.session_state:
    st.session_state.show_sources = True


def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>💼 TaxChat</h1>
        <p>Ask questions about Nigerian Tax Law</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ Please set OPENAI_API_KEY in your .env file")
        return
    
    # Load chatbot automatically
    with st.spinner("Loading..."):
        chatbot, error = init_chatbot()
    
    if error:
        st.error(f"❌ {error}")
        return
    
    # Toolbar: show sources toggle + clear button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.session_state.show_sources = st.toggle("📚 Show sources", value=st.session_state.show_sources)
    with col2:
        if st.button("🗑️ Clear"):
            st.session_state.messages = []
            chatbot.clear_history()
            st.rerun()
    
    st.divider()
    
    # Quick question buttons
    questions = [
        "💰 If I eran 250K per month, how much tax would I pay",
        "📊 Would I pay tax on my crypto gains?",
        "📝 Do I get rent deductions from tax?",
        "⛽ I own a farm, will I pay tax on profit of the farm?"
    ]
    
    cols = st.columns(len(questions))
    for col, q in zip(cols, questions):
        with col:
            if st.button(q, use_container_width=True):
                # Extract the actual question
                q_map = {
                    "💰 If I eran 250K per month, how much tax would I pay": "If I eran 250K per month, how much tax would I pay",
                    "📊 Would I pay tax on my crypto gains?": "Would I pay tax on my crypto gains?",
                    "📝 Do I get rent deductions from tax?": "Do I get rent deductions from tax?",
                    "⛽ I own a farm, will I pay tax on profit of the farm?": "I own a farm, will I pay tax on profit of the farm?"
                }
                st.session_state.pending_question = q_map[q]
                st.rerun()
    
    # Chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-msg">
                <div class="msg-label">You</div>
                <div class="msg-content">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bot-msg">
                <div class="msg-label">TaxChat</div>
                <div class="msg-content">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show sources
            if st.session_state.show_sources and msg.get("sources"):
                with st.expander("📚 Sources"):
                    for i, doc in enumerate(msg["sources"], 1):
                        title = doc.metadata.get('title', 'Untitled')
                        preview = doc.page_content[:200] + "..."
                        st.markdown(f"""
                        <div class="source-item">
                            <strong>#{i} {title}</strong><br>
                            <small>{preview}</small>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Handle pending question
    if hasattr(st.session_state, 'pending_question'):
        user_input = st.session_state.pending_question
        delattr(st.session_state, 'pending_question')
    else:
        user_input = st.chat_input("Ask a question...")
    
    # Process input
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("Thinking..."):
            try:
                result = chatbot.ask(user_input)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("context", [])
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Sorry, something went wrong: {e}",
                    "sources": None
                })
        
        st.rerun()


if __name__ == "__main__":
    main()
