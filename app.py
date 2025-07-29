import streamlit as st
import os
from typing import Optional, List
from phi.assistant import Assistant
from phi.storage.assistant.sqlite import SqliteAssistantStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.sqlite import SqliteVectorDb
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder
from phi.model.groq import Groq
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        display: flex;
        align-items: flex-start;
    }
    
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 2rem;
    }
    
    .stTextArea textarea {
        border-radius: 20px;
        border: 2px solid #667eea;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'assistant' not in st.session_state:
    st.session_state.assistant = None
if 'knowledge_base_loaded' not in st.session_state:
    st.session_state.knowledge_base_loaded = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = 'default_user'

@st.cache_resource
def initialize_knowledge_base(pdf_url: str):
    """Initialize and load the knowledge base"""
    try:
        db_file = "pdf_assistant_streamlit.db"
        
        knowledge_base = PDFUrlKnowledgeBase(
            urls=[pdf_url],
            vector_db=SqliteVectorDb(
                table_name="pdf_vectors",
                db_file=db_file,
                embedder=SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
            )
        )
        
        with st.spinner("Loading PDF and creating embeddings... This may take a few minutes."):
            knowledge_base.load()
        
        return knowledge_base, db_file
    except Exception as e:
        st.error(f"Error loading knowledge base: {str(e)}")
        return None, None

def initialize_assistant(knowledge_base, db_file, user_id: str):
    """Initialize the AI assistant"""
    try:
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            st.error("GROQ_API_KEY not found in environment variables!")
            return None
            
        storage = SqliteAssistantStorage(
            table_name='pdf_assistant_runs', 
            db_file=db_file
        )
        
        # Get existing run ID or create new one
        existing_run_ids = storage.get_all_run_ids(user_id)
        run_id = existing_run_ids[0] if existing_run_ids else None
        
        assistant = Assistant(
            run_id=run_id,
            user_id=user_id,
            knowledge_base=knowledge_base,
            storage=storage,
            model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
            show_tool_calls=True,
            search_knowledge=True,
            read_chat_history=True,
        )
        
        return assistant
    except Exception as e:
        st.error(f"Error initializing assistant: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 AI PDF Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # User settings
        user_id = st.text_input("User ID", value=st.session_state.current_user, help="Unique identifier for your chat session")
        if user_id != st.session_state.current_user:
            st.session_state.current_user = user_id
            st.session_state.assistant = None  # Reset assistant for new user
        
        # PDF URL input
        pdf_url = st.text_input(
            "PDF URL", 
            value="https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf",
            help="Enter the URL of the PDF you want to chat with"
        )
        
        # Load PDF button
        if st.button("🔄 Load PDF", type="primary"):
            if pdf_url:
                knowledge_base, db_file = initialize_knowledge_base(pdf_url)
                if knowledge_base:
                    st.session_state.knowledge_base = knowledge_base
                    st.session_state.db_file = db_file
                    st.session_state.knowledge_base_loaded = True
                    st.session_state.assistant = None  # Reset assistant for new PDF
                    st.success("PDF loaded successfully!")
                    st.rerun()
            else:
                st.error("Please enter a PDF URL")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.assistant = None
            st.success("Chat cleared!")
            st.rerun()
        
        # Status indicators
        st.subheader("📊 Status")
        if st.session_state.knowledge_base_loaded:
            st.markdown('<div class="success-box">✅ PDF Knowledge Base: Loaded</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-box">❌ PDF Knowledge Base: Not Loaded</div>', unsafe_allow_html=True)
        
        if st.session_state.assistant:
            st.markdown('<div class="success-box">✅ AI Assistant: Ready</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-box">❌ AI Assistant: Not Ready</div>', unsafe_allow_html=True)
        
        # Instructions
        st.subheader("📝 Instructions")
        st.markdown("""
        1. **Load PDF**: Enter a PDF URL and click "Load PDF"
        2. **Start Chatting**: Ask questions about the PDF content
        3. **Examples**:
           - "What is this document about?"
           - "Summarize the main points"
           - "Find information about [topic]"
        """)
    
    # Main chat interface
    if not st.session_state.knowledge_base_loaded:
        st.info("👈 Please load a PDF from the sidebar to start chatting!")
        
        # Quick start section
        st.subheader("🚀 Quick Start")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Features:**
            - 🤖 AI-powered PDF conversations
            - 🔍 Intelligent document search
            - 💾 Chat history persistence
            - 🎯 Context-aware responses
            """)
        
        with col2:
            st.markdown("""
            **Supported:**
            - 📄 PDF documents from URLs
            - 🌐 Public web-accessible files
            - 📚 Academic papers, manuals, reports
            - 📖 Books and documentation
            """)
        
        return
    
    # Initialize assistant if needed
    if not st.session_state.assistant and st.session_state.knowledge_base_loaded:
        st.session_state.assistant = initialize_assistant(
            st.session_state.knowledge_base, 
            st.session_state.db_file, 
            st.session_state.current_user
        )
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div>
                        <strong>👤 You:</strong><br>
                        {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <div>
                        <strong>🤖 Assistant:</strong><br>
                        {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    if st.session_state.assistant:
        # Use a form to handle enter key submission
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                user_input = st.text_area(
                    "Ask a question about the PDF:",
                    placeholder="Type your question here...",
                    height=100,
                    key="user_input"
                )
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
                submit_button = st.form_submit_button("Send 📤", type="primary")
        
        # Process user input
        if submit_button and user_input:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Get assistant response
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.assistant.run(user_input)
                    assistant_response = response.content if hasattr(response, 'content') else str(response)
                    
                    # Add assistant response to chat
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    
                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
            
            # Rerun to update the chat display
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        Built with ❤️ using Streamlit, Phi Framework, and Groq AI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()