# 🧠 MemAI — Cognitive Personal Assistant with Long-Term Memory

**MemAI** is a full-stack, personal cognitive assistant designed to eliminate context loss across chat sessions. By combining local vector search with high-speed LLM inference, MemAI automatically extracts personal facts, preferences, and goals from your conversations, indexes them in a persistent vector database, and dynamically injects relevant memories into future chats.

---

## 🌟 Key Features

* **Persistent Semantic Memory:** Extracts user facts automatically and indexes them into **Qdrant Vector DB** using local **FastEmbed (`all-MiniLM-L6-v2`)** 384-dimensional embeddings.
* **Ultra-Fast LLM Inference:** Powered by **Groq** (`openai/gpt-oss-20b`) for rapid response generation and entity extraction.
* **Multi-Session Conversation History:** Full conversation tracking with **SQLite**, automatic chat title generation, and pin/delete session management.
* **Modern Flutter UI:** Clean, responsive dark-mode mobile interface connected directly to the cloud backend.
* **User Data Isolation:** Queries and memories are strictly filtered by `user_id` to ensure clean separation of user data.

---

## 🏗️ Architecture & Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Mobile App** | Flutter (Dart) | Cross-platform mobile frontend |
| **Backend API** | FastAPI (Python) | High-performance asynchronous REST API |
| **LLM Inference** | Groq API (`openai/gpt-oss-20b`) | Fact extraction and prompt completion |
| **Vector Database** | Qdrant (Embedded) | Persistent 384-dimensional vector store |
| **Embedding Model** | FastEmbed (`all-MiniLM-L6-v2`) | Lightweight local embeddings (<100MB RAM) |
| **Relational Storage** | SQLite3 | Chat history, session management, and message logs |
| **Cloud Deployment** | Render | Automated web service hosting and CI/CD |

---

## 🔄 How It Works

[ User Prompt (Flutter) ]
│
├──► 1. FastEmbed Vectorization (384-dim) ──► Qdrant Vector Search ──► Relevant Context
│                                                                           │
├──► 2. SQLite History Retrieval (Last 10 turns) ───────────────────────────┤
│                                                                           │
└──► 3. Augmented Prompt Assembly ───────► Groq LLM Completion ◄────────────┘
│
├──► 4. Extract New Personal Facts
│         └──► Upsert to Qdrant DB
│
├──► 5. Commit Session to SQLite
│
└──► 6. Structured JSON to Client


---

## 📂 Project Structure

```text

├── backend/
│   ├── main.py              # FastAPI application, vector search & Groq logic
│   ├── requirements.txt     # Python dependencies
│   ├── render.yaml          # Cloud deployment configuration
│   └── local_qdrant_db/     # Embedded persistent vector database
│
└── frontend (memai)/
    ├── lib/
    │   ├── main.dart        # App entry point & theme configuration
    │   ├── screens/         # Chat screen, drawer & memory management
    │   ├── services/        # API service communicating with Render backend
    │   └── widgets/         # Custom message bubbles & components
    └── android/             # Android configuration & build files


