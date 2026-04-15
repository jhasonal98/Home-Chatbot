"""
Main Streamlit Application for Home Chatbot
Provides UI for hybrid interaction: room sections + Q&A chatbot powered by RAG
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    APP_TITLE, APP_DESCRIPTION, ROOM_SECTIONS, 
    SECTION_DESCRIPTIONS
)
from src.rag import get_rag_pipeline, retrieve_context
from src.llm import get_llm, answer_question


# ==============================================================================
# PAGE CONFIG & STYLING
# ==============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .response-box {
        background-color: #e8f4f8;
        border-left: 4px solid #0066cc;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .context-box {
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        padding: 0.8rem;
        border-radius: 5px;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    .room-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .room-section h2 {
        margin-top: 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ==============================================================================
# INITIALIZATION
# ==============================================================================

# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_section" not in st.session_state:
        st.session_state.current_section = "overview"
    if "rag_initialized" not in st.session_state:
        st.session_state.rag_initialized = False
    if "llm_initialized" not in st.session_state:
        st.session_state.llm_initialized = False


init_session_state()

# Initialize RAG and LLM with caching
@st.cache_resource
def init_rag():
    """Initialize RAG pipeline (cached for performance)."""
    try:
        pipeline = get_rag_pipeline()
        if pipeline.initialization_error:
            return pipeline, False, pipeline.initialization_error
        return pipeline, True, None
    except Exception as e:
        return None, False, str(e)


@st.cache_resource
def init_llm():
    """Initialize LLM (cached for performance)."""
    try:
        llm = get_llm()
        return llm, True
    except Exception as e:
        return None, False


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def render_section(section_key: str):
    """Render a room section with description."""
    section = SECTION_DESCRIPTIONS.get(section_key, {})
    
    st.markdown(f"""
        <div class="room-section">
            <h2>{section.get('title', 'Section')}</h2>
            <p>{section.get('content', 'No description available')}</p>
        </div>
    """, unsafe_allow_html=True)


def render_qa_interface():
    """Render the Q&A chat interface."""
    st.subheader("💬 Ask Questions About the Apartment")
    
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        if chat["role"] == "user":
            st.chat_message("user").write(chat["message"])
        else:
            with st.chat_message("assistant"):
                st.write(chat["message"])
                if chat.get("context"):
                    with st.expander("📚 View Sources"):
                        st.markdown("**Context Retrieved:**")
                        st.markdown(chat["context"])
    
    # Input area
    user_query = st.chat_input("Ask me anything about the apartment...")
    
    if user_query:
        # Display user message
        st.chat_message("user").write(user_query)
        
        # Get RAG pipeline
        pipeline, rag_ok, rag_error = init_rag()
        
        if not rag_ok:
            error_msg = rag_error or "Unknown error initializing RAG pipeline"
            st.error(f"❌ RAG Pipeline Error:\n\n{error_msg}\n\n**Troubleshooting:**\n- Check if sentence-transformers is installed\n- Delete `vectorstore/` folder and restart\n- Check README.md for full troubleshooting")
            return
        
        if pipeline is None or pipeline.vectorstore is None:
            st.error("❌ RAG system not ready. The embeddings model is still loading. Please wait a moment and try again.")
            return
        
        # Retrieve context from RAG
        try:
            with st.spinner("🔍 Searching apartment knowledge base..."):
                context_chunks, metadata = retrieve_context(user_query, k=3)
                context_string = "\n\n---\n\n".join(context_chunks) if context_chunks else ""
        except Exception as e:
            st.error(f"❌ Error during retrieval: {str(e)}")
            return
        
        # Generate answer using LLM
        llm, llm_ok = init_llm()
        
        if not llm_ok or llm is None:
            st.warning(
                "⚠️ LLM is not available. Please add your HuggingFace API token to `.streamlit/secrets.toml`\n\n"
                "Steps:\n"
                "1. Get a free token from https://huggingface.co/settings/tokens\n"
                "2. Add to `.streamlit/secrets.toml`: `HF_API_TOKEN = 'your_token_here'`\n"
                "3. Restart the app"
            )
            response = "LLM not available. Please configure your HuggingFace API token."
        else:
            try:
                with st.spinner("🤖 Generating response..."):
                    response = answer_question(user_query, context_string)
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                return
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(response)
            if context_chunks:
                with st.expander("📚 View Sources"):
                    st.markdown("**Retrieved Context:**")
                    for i, chunk in enumerate(context_chunks, 1):
                        st.markdown(f"**Source {i}:**\n{chunk}")
        
        # Store in chat history
        st.session_state.chat_history.append({
            "role": "user",
            "message": user_query
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "message": response,
            "context": "\n\n---\n\n".join(context_chunks) if context_chunks else None
        })


# ==============================================================================
# MAIN LAYOUT
# ==============================================================================

def main():
    """Main application layout."""
    
    # Header
    st.title(APP_TITLE)
    st.markdown(f"*{APP_DESCRIPTION}*")
    st.markdown("---")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("📑 Navigation")
        selected_section = st.radio(
            "Choose a section:",
            options=list(ROOM_SECTIONS.keys()),
            format_func=lambda x: x,
            key="section_selector"
        )
        
        st.markdown("---")
        st.markdown("### About This App")
        st.markdown("""
        **Hybrid Interface:**
        - **Room Sections:** View pre-written descriptions of each room
        - **Q&A Chat:** Ask custom questions powered by RAG
        
        **Technology:**
        - 🤖 LLM: HuggingFace Inference API
        - 📚 Vector DB: FAISS (local)
        - 🔍 Embeddings: Sentence Transformers
        - 🎨 UI: Streamlit
        """)
        
        st.markdown("---")
        st.info(
            "💡 **Tip:** For the best experience, use the Q&A section to ask specific questions about features you're interested in!"
        )
    
    # Main content area based on selected section
    section_key = ROOM_SECTIONS.get(selected_section, "overview")
    
    if section_key == "qa_chat":
        render_qa_interface()
    else:
        render_section(section_key)
        
        # Add a Q&A prompt at the bottom of each section
        st.markdown("---")
        st.markdown("### 🤔 Have a specific question?")
        st.markdown("Switch to the **💬 Q&A Chat** tab to ask anything about the apartment!")


if __name__ == "__main__":
    main()
