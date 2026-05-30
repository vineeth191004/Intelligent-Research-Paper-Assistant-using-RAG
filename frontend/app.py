import os
import tempfile
import base64
import streamlit as st
import pandas as pd
from config import settings
from ingestion.parser import process_pdf
from retrieval.search import HybridSearchEngine
from generation.llm import OllamaGenerator

# App setup
st.set_page_config(
    page_title="Aurora AI - Research Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Helper function to encode image for HTML
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
    robot_b64 = get_base64_of_bin_file("frontend/assets/robot.png")
except Exception:
    robot_b64 = ""

# Custom Styling (Dark Glassmorphism UI)
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Outfit', sans-serif;
    }}
    
    .stApp {{
        background-color: #0d1117;
        color: #c9d1d9;
    }}
    
    /* Hero Section */
    .hero-section {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 2rem;
        background: linear-gradient(135deg, rgba(31,41,55,0.7) 0%, rgba(17,24,39,0.7) 100%);
        border: 1px solid #374151;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    }}
    
    .hero-image-container {{
        flex-shrink: 0;
        animation: float 6s ease-in-out infinite;
    }}
    
    @keyframes float {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-10px); }}
        100% {{ transform: translateY(0px); }}
    }}
    
    .hero-image {{
        width: 180px;
        height: 180px;
        border-radius: 50%;
        object-fit: cover;
        box-shadow: 0 0 20px rgba(96, 165, 250, 0.4);
        border: 2px solid #3b82f6;
    }}
    
    .hero-text h1 {{
        background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        font-size: 2.8rem;
        line-height: 1.2;
    }}
    
    .hero-text p {{
        color: #9ca3af !important;
        font-size: 1.1rem;
        margin: 0;
        line-height: 1.5;
    }}
    
    /* Citation cards */
    .citation-card {{
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #60a5fa;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}
    
    .citation-header {{
        display: flex;
        justify-content: space-between;
        font-weight: 600;
        color: #60a5fa;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }}
    
    .citation-text {{
        font-size: 0.88rem;
        color: #cbd5e1;
        line-height: 1.5;
        font-style: italic;
    }}
    
    .score-badge {{
        background: rgba(99, 102, 241, 0.2);
        color: #a78bfa;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        border: 1px solid rgba(167, 139, 250, 0.3);
    }}

    /* Side details */
    .file-pill {{
        background-color: #1e293b;
        color: #cbd5e1;
        padding: 0.3rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        border: 1px solid #334155;
        margin: 0.2rem;
        display: inline-block;
    }}
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "search_engine" not in st.session_state:
    st.session_state.search_engine = HybridSearchEngine()
    success = st.session_state.search_engine.load_indices()
    st.session_state.db_loaded = success
else:
    success = st.session_state.search_engine.load_indices()
    st.session_state.db_loaded = success

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
    if st.session_state.db_loaded:
        import json
        if os.path.exists(settings.CHUNKS_METADATA_PATH):
            try:
                with open(settings.CHUNKS_METADATA_PATH, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    st.session_state.processed_files = list(set(c["metadata"]["source"] for c in chunks))
            except Exception:
                pass

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=70)
    st.title("Settings")

    st.subheader("🦙 Ollama Local LLM")
    ollama_ok = settings.is_ollama_running()
    if ollama_ok:
        st.success("✅ Ollama is running.")
    else:
        st.error("❌ Ollama not found.")
        st.markdown(f"Ensure Ollama is running at `{settings.OLLAMA_BASE_URL}`")
        
    model_name_input = st.text_input(
        "Ollama Model",
        value=settings.OLLAMA_MODEL_NAME,
        help="Type the name of your local Ollama model (e.g. 'llama3', 'llama3.2', 'mistral')"
    )
    settings.OLLAMA_MODEL_NAME = model_name_input

    st.markdown("---")
    st.subheader("RAG Parameters")

    chunk_size = st.slider("Chunk Size (characters)", 500, 2000, settings.CHUNK_SIZE, 100)
    chunk_overlap = st.slider("Chunk Overlap (characters)", 50, 500, settings.CHUNK_OVERLAP, 50)
    top_k_retrieve = st.slider("Retrieval candidates (K)", 5, 30, settings.TOP_K_RETRIEVAL, 5)
    top_k_rerank = st.slider("Final contexts (N)", 1, 10, settings.TOP_K_RERANK, 1)
    
    st.markdown("---")
    st.subheader("Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload research papers (PDFs)", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process & Index Documents", use_container_width=True):  # noqa: streamlit
            all_chunks = []
            progress_bar = st.progress(0, text="Reading files...")
            
            for idx, uploaded_file in enumerate(uploaded_files):
                progress_bar.progress((idx) / len(uploaded_files), text=f"Parsing {uploaded_file.name}...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                try:
                    chunks = process_pdf(tmp_path)
                    for chunk in chunks:
                        chunk["metadata"]["source"] = uploaded_file.name
                    all_chunks.extend(chunks)
                finally:
                    os.remove(tmp_path)
            
            if all_chunks:
                progress_bar.progress(0.8, text="Generating embeddings and building indices (BM25 + FAISS)...")
                settings.CHUNK_SIZE = chunk_size
                settings.CHUNK_OVERLAP = chunk_overlap
                
                st.session_state.search_engine.index_documents(all_chunks)
                st.session_state.db_loaded = True
                st.session_state.processed_files = [f.name for f in uploaded_files]
                progress_bar.progress(1.0, text="Indexing complete!")
                st.success(f"Successfully processed {len(uploaded_files)} PDF(s) into {len(all_chunks)} semantic chunks!")
            else:
                st.error("Could not extract any text from the uploaded PDFs.")
                
    st.markdown("---")
    if st.button("Clear Vector Store & History", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.processed_files = []
        st.session_state.db_loaded = False
        
        for p in [settings.FAISS_INDEX_PATH, settings.BM25_INDEX_PATH, settings.CHUNKS_METADATA_PATH]:
            if os.path.exists(p):
                os.remove(p)
        st.rerun()

# Main Layout
st.markdown(f"""
<div class="hero-section">
    <div class="hero-image-container">
        <img src="data:image/png;base64,{robot_b64}" class="hero-image" alt="AI Robot Avatar">
    </div>
    <div class="hero-text">
        <h1>Hi, I'm Aurora.</h1>
        <p>I am your highly advanced machine assistant. Upload your lengthy research papers, manuals, or documents in the sidebar, and I'll read them in seconds to answer any questions you have.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Chat Interface
if st.session_state.processed_files:
    st.markdown("**Indexed Documents:**")
    for f in st.session_state.processed_files:
        st.markdown(f'<span class="file-pill">📄 {f}</span>', unsafe_allow_html=True)
    st.write("")
else:
    st.warning("No documents have been indexed yet. Please upload research papers in the sidebar.")

# Initialize generator
generator = OllamaGenerator()

# Display Chat Messages
for msg in st.session_state.messages:
    # Use a custom robot emoji for the assistant
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        
        # Show citations if present
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander("🔍 Citations & Retrieved Contexts"):
                for idx, cit in enumerate(msg["citations"], start=1):
                    st.markdown(f"""
                    <div class="citation-card">
                        <div class="citation-header">
                            <span>[{idx}] Page {cit['page']} — {cit['source']}</span>
                            <span class="score-badge">Rerank Score: {cit['rerank_score']:.4f}</span>
                        </div>
                        <div class="citation-text">"{cit['text']}"</div>
                    </div>
                    """, unsafe_allow_html=True)

# Chat Input
if query := st.chat_input("Ask Aurora a question about the papers...", disabled=not st.session_state.db_loaded):
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 *Reconstructing query and performing hybrid search...*")

        # 2. Rephrase follow up based on memory
        turns = []
        for m in st.session_state.messages[:-1]:
            if m["role"] == "user":
                turns.append({"user": m["content"], "assistant": ""})
            elif m["role"] == "assistant" and len(turns) > 0:
                turns[-1]["assistant"] = m["content"]

        standalone_query = generator.condense_query(turns, query)

        # 3. Hybrid search retrieval
        retrieved_chunks = st.session_state.search_engine.search(
            standalone_query,
            top_k_retrieve=top_k_retrieve,
            top_k_rerank=top_k_rerank
        )

        # 4. Generate answer
        message_placeholder.markdown("🧠 *Generating answer using Ollama...*")
        response = generator.generate_answer(query, retrieved_chunks, turns)

        answer = response["answer"]
        citations = response["citations"]

        message_placeholder.markdown(answer)

        # Show citations dropdown
        if citations:
            with st.expander("🔍 Citations & Retrieved Contexts"):
                for idx, cit in enumerate(citations, start=1):
                    st.markdown(f"""
                    <div class="citation-card">
                        <div class="citation-header">
                            <span>[{idx}] Page {cit['page']} — {cit['source']}</span>
                            <span class="score-badge">Rerank Score: {cit['rerank_score']:.4f}</span>
                        </div>
                        <div class="citation-text">"{cit['text']}"</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Store assistant response in session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "citations": citations
        })
