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
    page_title="TaxChat - Nigerian Tax Law Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #58a6ff;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #8b949e;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        background-color: #161b22;
        color: #c9d1d9;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        color: #c9d1d9;
    }
    .user-message {
        background-color: #0d1117;
        border-left: 4px solid #58a6ff;
    }
    .assistant-message {
        background-color: #161b22;
        border-left: 4px solid #3fb950;
    }
    .source-box {
        background-color: #1c1f24;
        border: 1px solid #e3b341;
        border-radius: 0.3rem;
        padding: 0.8rem;
        margin-top: 0.5rem;
        color: #e3b341;
    }
    .info-box {
        background-color: #0d1117;
        border: 1px solid #388bfd;
        border-radius: 0.3rem;
        padding: 1rem;
        margin-bottom: 1rem;
        color: #c9d1d9;
    }
</style>
""", unsafe_allow_html=True)
# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

if "vectorstore_loaded" not in st.session_state:
    st.session_state.vectorstore_loaded = False

if "show_sources" not in st.session_state:
    st.session_state.show_sources = True


def load_rag_system():
    """Load the RAG system with the vectorstore."""
    try:
        with st.spinner("🔄 Loading knowledge base..."):
            vectorstore = load_vectorstore(db_name=db_name)
            chatbot = create_rag_chatbot(vectorstore, model_name=MODEL, temperature=0.7)
            st.session_state.chatbot = chatbot
            st.session_state.vectorstore_loaded = True
            return True
    except FileNotFoundError:
        st.error("❌ Vectorstore not found! Please run main.py first to create the knowledge base.")
        return False
    except Exception as e:
        st.error(f"❌ Error loading RAG system: {e}")
        return False


def display_message(role, content, sources=None):
    """Display a chat message with optional sources."""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🤔 You:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 Assistant:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
        
        # Display sources if available and enabled
        if sources and st.session_state.show_sources:
            with st.expander("📚 View Sources", expanded=False):
                for i, doc in enumerate(sources, 1):
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Source {i}: {doc.metadata.get('title', 'Untitled')}</strong><br>
                        <em>Summary:</em> {doc.metadata.get('summary', 'N/A')[:200]}...<br>
                        <em>Content Preview:</em> {doc.page_content[:300]}...
                    </div>
                    """, unsafe_allow_html=True)


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">💼 TaxChat</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your AI Assistant for Nigerian Tax Law</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.warning("⚠️ OPENAI_API_KEY not found in environment variables!")
            api_key = st.text_input("Enter your OpenAI API Key:", type="password")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
        else:
            st.success("✅ API Key loaded")
        
        st.divider()
        
        # Load vectorstore button
        if not st.session_state.vectorstore_loaded:
            if st.button("🚀 Load Knowledge Base", use_container_width=True):
                if load_rag_system():
                    st.success("✅ Knowledge base loaded successfully!")
                    st.rerun()
        else:
            st.success("✅ Knowledge base ready")
            
            # Settings
            st.session_state.show_sources = st.checkbox(
                "Show sources",
                value=st.session_state.show_sources,
                help="Display source documents used to generate answers"
            )
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.chatbot:
                    st.session_state.chatbot.clear_history()
                st.rerun()
        
        st.divider()
        
        # Information
        st.markdown("""
        <div class="info-box">
            <strong>📖 About TaxChat</strong><br>
            This chatbot helps you understand the Nigerian Tax Act 2025. 
            Ask questions about tax laws, regulations, and procedures.
        </div>
        """, unsafe_allow_html=True)
        
        # Example questions
        st.markdown("### 💡 Example Questions")
        example_questions = [
            "What is the tax rate for companies?",
            "How is personal income tax calculated?",
            "What are the deductions allowed?",
            "Explain petroleum profits tax",
            "What is stamp duty?"
        ]
        
        for question in example_questions:
            if st.button(f"📝 {question}", key=question, use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()
    
    # Main chat area
    if not st.session_state.vectorstore_loaded:
        st.info("👈 Please load the knowledge base from the sidebar to start chatting.")
        st.markdown("""
        ### How to use:
        1. Click "Load Knowledge Base" in the sidebar
        2. Wait for the system to initialize
        3. Start asking questions about Nigerian tax law
        
        ### Features:
        - 💬 Natural conversation with context awareness
        - 📚 Source citations for all answers
        - 🔍 Search through tax documents instantly
        - 🧠 Powered by GPT-4o-mini and advanced RAG
        """)
        return
    
    # Display chat history
    for message in st.session_state.messages:
        display_message(
            message["role"],
            message["content"],
            message.get("sources")
        )
    
    # Handle pending question from example buttons
    if hasattr(st.session_state, 'pending_question'):
        user_input = st.session_state.pending_question
        delattr(st.session_state, 'pending_question')
    else:
        # Chat input
        user_input = st.chat_input("Ask a question about Nigerian tax law...")
    
    if user_input:
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Get response from chatbot
        with st.spinner("🤔 Thinking..."):
            try:
                result = st.session_state.chatbot.ask(user_input)
                
                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("context", [])
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error: {e}",
                    "sources": None
                })


if __name__ == "__main__":
    main()

