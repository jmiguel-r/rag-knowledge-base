# 🤖 Project 1 — Agentic Knowledge Base Assistant (RAG System)

> **Agentic AI Bootcamp** · Built by J. Miguel Ramírez  
> Instructor: Satyajit Pattnaik — Lead AI and Data Consultant at PALO IT

---

## 📌 Project Overview

A **RAG-powered chatbot** that answers questions from custom web-based documents by combining intelligent retrieval with generative AI. This project showcases applied skills in:

- **Retrieval-Augmented Generation (RAG)**
- **Context-aware response generation**
- **Applied Natural Language Processing (NLP)**
- **Local LLM deployment with Ollama**

---

## 🏗️ Architecture

```
Web URLs
   │
   ▼
[WebBaseLoader]                  ← Fetches and parses HTML content
   │
   ▼
[RecursiveCharacterTextSplitter] ← Splits text into 1000-char chunks (150 overlap)
   │
   ▼
[OllamaEmbeddings]               ← Converts chunks to semantic vectors
   │  (nomic-embed-text)
   ▼
[ChromaDB]                       ← Persists vectors locally in ./chroma_db
   │
   │   ── at query time ──
   ▼
[Similarity Retriever (k=4)]     ← Finds 4 most relevant chunks
   │
   ▼
[ChatPromptTemplate]             ← Injects context into the system prompt
   │
   ▼
[ChatOllama / llama3.2]          ← Generates a grounded answer
   │
   ▼
Answer + Sources
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Framework** | LangChain 0.3 | RAG orchestration via LCEL |
| **Vector Store** | ChromaDB | Local persistent embeddings storage |
| **LLM Runtime** | Ollama | Local model execution (no API cost) |
| **Embedding Model** | `nomic-embed-text` | Semantic text vectorization |
| **Chat Model** | `llama3.2` | Answer generation |
| **Document Loader** | WebBaseLoader | Web scraping and parsing |
| **Text Splitter** | RecursiveCharacterTextSplitter | Intelligent chunking |
| **Terminal UI** | Rich | Colored, formatted CLI output |
| **Language** | Python 3.10+ | Core implementation |

---

## 🚀 v2 — Agentic RAG with LangGraph

### v1 vs v2 Comparison

| Aspect | v1 — `rag.py` | v2 — `agent.py` |
|---|---|---|
| **Orchestration** | LangChain LCEL (linear chain) | LangGraph StateGraph (cyclic graph) |
| **Retrieval** | Single-shot, no validation | Graded + retry loop (max 3 iterations) |
| **Query** | Fixed, as typed | Rewritten by LLM on each retry |
| **Document Quality** | All retrieved chunks used | Only LLM-graded relevant chunks kept |
| **Failure Handling** | Returns whatever was retrieved | Retries with an improved query up to 3× |
| **State** | Implicit (chain variables) | Explicit `AgentState` TypedDict |
| **Observability** | Status spinner | Step-by-step node execution logging |

### Graph Flow

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │
                         ▼
                   ┌───────────┐
                   │  retrieve │  ← ChromaDB similarity search (k=4)
                   └─────┬─────┘
                         │
                         ▼
               ┌──────────────────┐
               │ grade_documents  │  ← LLM grades each chunk: yes / no
               └────────┬─────────┘
                        │
            ┌───────────┴────────────┐
            │     should_retry()     │
            └──┬──────────────────┬──┘
  not relevant │  iterations < 3  │  relevant  OR  iterations ≥ 3
               ▼                  ▼
      ┌────────────────┐    ┌──────────┐
      │ rewrite_query  │    │ generate │  ← Grounded LLM answer
      └───────┬────────┘    └────┬─────┘
              │  (loop back)     │
              └──► retrieve      ▼
                                END
```

### Run v2

```bash
python agent.py
```

Type your questions as normal. The agent will log each pipeline step — retrieval, grading, and optional rewrites — before printing the final answer with source attribution.

### Key Concepts Demonstrated

| Concept | Implementation |
|---|---|
| **LangGraph StateGraph** | Cyclic graph with typed state, conditional edges, and explicit transitions |
| **Adaptive Retrieval** | Query rewriting loop that improves recall on unsuccessful retrievals |
| **LLM-as-Judge** | Dedicated `grade_documents` node scores each chunk individually |
| **Bounded Iteration** | `MAX_ITER = 3` prevents infinite retry loops |
| **Explicit State** | `AgentState` TypedDict makes every agent decision inspectable at runtime |

---

## 🤖 Models Used

### `nomic-embed-text` — Embeddings
- Converts text chunks into high-dimensional semantic vectors
- Runs locally via Ollama (~274 MB)
- Used in both `ingest.py` and `rag.py` (must be consistent)

### `llama3.2` — Chat / Generation
- Meta's open-source LLM for text generation
- Receives retrieved context + user question → generates grounded answer
- Runs locally via Ollama (~2 GB)
- `temperature=0` for deterministic, factual responses

---

## 📁 Project Structure

```
rag-knowledge-base/
├── ingest.py          # Ingestion pipeline: load → split → embed → store
├── rag.py             # Query engine: retrieve → prompt → generate → chat
├── requirements.txt   # Python dependencies
├── .gitignore         # Excludes venv, chroma_db, .env
├── .env.example       # Template for environment variables (OpenAI variant)
└── README.md          # This file
```

> **Note:** `chroma_db/` is auto-generated by `ingest.py` and excluded from Git.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### 1. Clone the repository
```bash
git clone https://github.com/your-username/rag-knowledge-base.git
cd rag-knowledge-base
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Pull required Ollama models
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 5. Verify Ollama is running
```bash
ollama ps
ollama list
```

---

## 🚀 Usage

### Step 1 — Ingest documents (run once, or when URLs change)
```bash
python ingest.py
```
This will:
- Download and parse the pages in `URLS[]`
- Split content into 1000-character chunks with 150-char overlap
- Generate embeddings using `nomic-embed-text`
- Save the vector store to `./chroma_db`

### Step 2 — Start the chatbot
```bash
python rag.py
```
Type your questions and get answers grounded in your knowledge base. Type `exit` to quit.

---

## 🔧 Customization

| Parameter | File | Description |
|---|---|---|
| `URLS` | `ingest.py` | Web pages to index |
| `chunk_size` | `ingest.py` | Characters per chunk (default: 1000) |
| `chunk_overlap` | `ingest.py` | Overlap between chunks (default: 150) |
| `k=4` | `rag.py` | Number of chunks retrieved per query |
| `SYSTEM_PROMPT` | `rag.py` | Assistant personality and instructions |
| `LLM_MODEL` | `rag.py` | Swap `llama3.2` for any Ollama model |

---

## 🔄 Variants

This project was developed in two variants:

| Variant | LLM | Embeddings | Cost |
|---|---|---|---|
| **Ollama (this repo)** | llama3.2 | nomic-embed-text | Free / Local |
| **OpenAI** | gpt-4o-mini | text-embedding-3-small | Pay-per-token |

---

## 🗺️ Roadmap

- [ ] Add conversation memory (`ConversationBufferMemory`)
- [ ] Migrate to **LangGraph** for agentic flows
- [ ] Add **Streamlit** UI
- [ ] Add **LangSmith** for observability and tracing
- [ ] Support PDF ingestion alongside URLs

---

## 📚 Bootcamp Context

This project is part of the **Building Agentic AI Applications** Bootcamp:

- Beginner-friendly · Interactive live sessions · Recordings available
- Topics covered: Python, LLMs, RAG, LangChain, CrewAI, LangGraph, no-code agents
- Projects: Document Q&A Chatbot · Smart Weather Assistant · Travel Planner · Resume Screening · Text-to-SQL Chatbot

---

## 👤 Author

**J. Miguel Ramírez**  
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)
