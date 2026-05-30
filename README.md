# Intelligent Research Paper Assistant using Hybrid RAG

A production-grade, modular Retrieval-Augmented Generation (RAG) system for question answering over research papers (PDFs). This project integrates advanced search retrieval and LLM evaluation telemetry to demonstrate an industry-standard NLP solution suitable for AI/ML engineering portfolios.

---

## 🚀 Resume Description

> **Research Paper Question Answering System using RAG and LLMs**
> *Developed a Retrieval-Augmented Generation (RAG) chatbot for research paper question answering using LangChain, FAISS, Sentence Transformers, and Ollama. Implemented hybrid retrieval, semantic search, citation-based responses, and conversational memory, enabling accurate document-grounded answers from uploaded PDFs.*

---

## 🏗️ Architecture

The system follows a clean, modular pipeline designed to run efficiently on standard CPU-based instances:

```text
       📄 Multi-PDF Upload
               |
               v
       📝 Ingestion Engine [PyPDF extraction, cleaning, chunking]
               |
               +----------------------+----------------------+
               |                                             |
               v (Sparse Path)                               v (Dense Path)
       🔤 BM25 Tokenizer                       🧠 SentenceTransformer Embeddings
               |                                             |
               v                                             v
       🗄️ BM25 Index (Keyword Search)          📂 FAISS Index (Semantic Search)
               |                                             |
               +----------------------+----------------------+
                                      |
                                      v
                        🔀 Reciprocal Rank Fusion (RRF)
                                      |
                                      v
                        🧬 Cross-Encoder Reranker (MS-Marco)
                                      | (Top-N contexts)
                                      v
                        💬 Ollama Local LLM Answer Generation
                               [Grounded with page-level citations]
```

---

## 🌟 Key Features

1. **Multi-Document Ingestion**: Upload and index multiple PDF files (e.g. arXiv papers, corporate reports, technical guides) concurrently.
2. **Hybrid Search (Sparse + Dense)**: Combines keyword-matching (**BM25**) and semantic vector-matching (**FAISS** with `all-MiniLM-L6-v2`) to capture exact terms and thematic context.
3. **Reciprocal Rank Fusion (RRF)**: Merges sparse and dense search results using rank-based reciprocal scaling ($k=60$), ensuring balanced retrieval scores.
4. **Cross-Encoder Reranking**: Re-evaluates top retrieved chunks using a `cross-encoder/ms-marco-MiniLM-L-6-v2` model to maximize chunk relevance.
5. **Conversational Memory**: Implements contextual query condensation. Chat history and current queries are rephrased into a standalone query before vector search.
6. **Page-level Citations**: Highlights specific source files and page numbers for every generated claim, minimizing LLM hallucinations.
---

## 🛠️ Technology Stack

- **ML / AI**: LangChain, Sentence Transformers (`all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`), FAISS Vector Store, Ollama.
- **Backend & Parser**: Python, PyPDF, rank_bm25.
- **Frontend & Visuals**: Streamlit, Pandas.

---

## 📁 Repository Structure

```text
rag chatbot/
├── config/
│   └── settings.py           # Configuration values, model names, and file paths
├── ingestion/
│   └── parser.py             # PDF parsers and character text splitter
├── embeddings/
│   └── embedder.py           # Local SentenceTransformers embedder CPU wrapper
├── retrieval/
│   ├── bm25_retriever.py     # Sparse BM25 retrieval index
│   ├── faiss_retriever.py    # Dense FAISS cosine similarity vector store
│   └── search.py             # RRF Hybrid search and Cross-Encoder reranker
├── generation/
│   └── llm.py                # Conversational history re-writer and Ollama QA generator
├── evaluation/
│   └── metrics.py            # Precision, Recall, and LLM-assisted RAG metrics
├── frontend/
│   └── app.py                # Streamlit UI, chat views, and evaluation benches
├── tests/
│   └── test_rag.py           # Automated unit test suite
├── requirements.txt          # Package dependencies
└── run.sh                    # Startup wrapper script
```

---

## ⚙️ Setup & Deployment

### Prerequisites
- Python 3.10 or higher
- **Ollama** installed on your system with the `llama3` model downloaded (`ollama run llama3`)

### Installation
1. Clone the project to your local workspace.
2. Initialize a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install package dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Execution
1. Ensure Ollama is running in the background.
2. Start the Streamlit application:
   ```bash
   ./run.sh
   ```
3. Open `http://localhost:8501` in your browser.

---

## 🧪 Testing

Run the automated unit tests to verify parser cleaning, BM25 indices, Reciprocal Rank Fusion logic, and evaluation calculations:
```bash
python3 -m unittest tests/test_rag.py
```
