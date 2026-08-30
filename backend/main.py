import os
import re
import json
import uuid
import sqlite3
import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from fastembed import TextEmbedding

# ------------------------------------------------------------
# ENVIRONMENT & LOGGING
# ------------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing! Please check your .env file or Render environment settings.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------------
app = FastAPI(title="MemAI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# SQLITE SESSION & CONVERSATION DATABASE
# ------------------------------------------------------------
DB_FILE = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            is_pinned INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            sender TEXT NOT NULL,
            text TEXT NOT NULL,
            recalled_memories TEXT,
            saved_memories TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ------------------------------------------------------------
# INITIALIZATION (GROQ, QDRANT, FASTEMBED)
# ------------------------------------------------------------
MODEL_ID = "openai/gpt-oss-20b"
groq_client = Groq(api_key=GROQ_API_KEY)

# Lightweight embedding model (<100MB RAM footprint)
embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text: str) -> List[float]:
    """Generates a 384-dimensional vector embedding using FastEmbed."""
    embeddings = list(embedder.embed([text]))
    return embeddings[0].tolist()

qdrant_client = QdrantClient(path="./local_qdrant_db")
COLLECTION_NAME = "user_memories"

existing_cols = [c.name for c in qdrant_client.get_collections().collections]
if COLLECTION_NAME not in existing_cols:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE),
    )

# ------------------------------------------------------------
# PYDANTIC SCHEMAS
# ------------------------------------------------------------
class ChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    session_title: str
    recalled_memories: List[str]
    saved_memories: List[str] = []


class MemoryItem(BaseModel):
    id: str
    memory: str
    created_at: Optional[str] = None


class SessionItem(BaseModel):
    id: str
    user_id: str
    title: str
    is_pinned: bool
    created_at: str
    updated_at: str


class MessageItem(BaseModel):
    id: str
    session_id: str
    sender: str
    text: str
    recalled_memories: List[str] = []
    saved_memories: List[str] = []
    created_at: str


class CreateSessionRequest(BaseModel):
    user_id: str
    title: Optional[str] = "New Chat"


class UpdatePinRequest(BaseModel):
    is_pinned: bool


class UpdateTitleRequest(BaseModel):
    title: str


# ------------------------------------------------------------
# FACT EXTRACTION ENGINE
# ------------------------------------------------------------
def extract_and_save_facts(user_id: str, user_text: str) -> List[str]:
    saved_facts = []
    try:
        extract_prompt = f"""Extract all factual statements, ambitions, tasks, relationships, and preferences from the user statement.
Format strictly as a bulleted list starting with '-'.
If the message is only a generic greeting or simple question with no personal info, return nothing.

User input:
"{user_text}"

Extracted facts:"""

        resp = groq_client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": extract_prompt}],
            temperature=0.0,
            max_tokens=250,
        )
        raw_output = resp.choices[0].message.content.strip()

        extracted_lines = []
        for line in raw_output.split("\n"):
            cleaned = re.sub(r"^[\*\-\d\.\s]+", "", line).strip()
            if cleaned and len(cleaned) > 2 and not cleaned.lower().startswith("extracted"):
                extracted_lines.append(cleaned)

        for fact_str in extracted_lines:
            fact_vec = get_embedding(fact_str)
            point_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    qmodels.PointStruct(
                        id=point_id,
                        vector=fact_vec,
                        payload={
                            "user_id": user_id,
                            "memory": fact_str,
                            "created_at": timestamp,
                        },
                    )
                ],
            )
            saved_facts.append(fact_str)

        logger.info(f"[{user_id}] Stored memories: {saved_facts}")
    except Exception as e:
        logger.warning(f"Fact extraction warning: {e}")

    return saved_facts


def generate_session_title(first_message: str) -> str:
    """Generates a concise 3-5 word title for a new chat session."""
    try:
        resp = groq_client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": "Create a 3 to 5 word title summarizing the user's initial message. Return ONLY the title text.",
                },
                {"role": "user", "content": first_message},
            ],
            temperature=0.3,
            max_tokens=20,
        )
        title = resp.choices[0].message.content.strip().replace('"', '').replace("'", "")
        return title if title else first_message[:30]
    except Exception:
        return first_message[:30]


# ------------------------------------------------------------
# CHAT ENDPOINT
# ------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # 1. Ensure or create active session
        session_id = payload.session_id
        session_title = "New Chat"

        if not session_id:
            session_id = f"sess_{uuid.uuid4().hex[:10]}"
            session_title = generate_session_title(payload.message)
            cursor.execute(
                "INSERT INTO sessions (id, user_id, title, is_pinned, created_at, updated_at) VALUES (?, ?, ?, 0, ?, ?)",
                (session_id, payload.user_id, session_title, now_iso, now_iso),
            )
        else:
            cursor.execute("SELECT title FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            if row:
                session_title = row[0]
                cursor.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now_iso, session_id))
            else:
                session_title = generate_session_title(payload.message)
                cursor.execute(
                    "INSERT INTO sessions (id, user_id, title, is_pinned, created_at, updated_at) VALUES (?, ?, ?, 0, ?, ?)",
                    (session_id, payload.user_id, session_title, now_iso, now_iso),
                )

        # 2. Retrieve global user memory from Qdrant
        query_vec = get_embedding(payload.message)
        user_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="user_id",
                    match=qmodels.MatchValue(value=payload.user_id),
                )
            ]
        )

        recalled_texts = []
        if hasattr(qdrant_client, "query_points"):
            search_res = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vec,
                query_filter=user_filter,
                limit=10,
            )
            recalled_texts = [
                hit.payload.get("memory")
                for hit in search_res.points
                if hit.payload and hit.payload.get("memory")
            ]
        else:
            search_res = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vec,
                query_filter=user_filter,
                limit=10,
            )
            recalled_texts = [
                hit.payload.get("memory")
                for hit in search_res
                if hit.payload and hit.payload.get("memory")
            ]

        memory_context = (
            "\n".join([f"- {m}" for m in recalled_texts])
            if recalled_texts
            else "None recorded yet."
        )

        # 3. Retrieve recent conversation history for this session
        cursor.execute(
            "SELECT sender, text FROM messages WHERE session_id = ? ORDER BY created_at ASC LIMIT 10",
            (session_id,),
        )
        past_msgs = cursor.fetchall()

        # Build prompt payload
        messages_payload = [
            {
                "role": "system",
                "content": f"""You are MemAI, an executive-tier, highly intelligent cognitive assistant.

Persistent User Knowledge Bank:
{memory_context}

Communication Guidelines:
- Deliver direct, high-value, and actionable insights.
- Eliminate generic conversational fluff and boilerplate closings.
- Structure responses using concise bold categories and clean bullet points.
- Proactively leverage stored facts to personalize your guidance.""",
            }
        ]

        for sender, text in past_msgs:
            messages_payload.append({
                "role": "user" if sender == "user" else "assistant",
                "content": text,
            })

        messages_payload.append({"role": "user", "content": payload.message})

        # 4. LLM Completion
        completion = groq_client.chat.completions.create(
            model=MODEL_ID,
            messages=messages_payload,
            temperature=0.4,
            max_tokens=600,
        )
        ai_reply = completion.choices[0].message.content.strip()

        # 5. Extract and save new facts
        saved_facts = extract_and_save_facts(payload.user_id, payload.message)

        # 6. Save message history to SQLite
        user_msg_id = f"msg_{uuid.uuid4().hex[:10]}"
        ai_msg_id = f"msg_{uuid.uuid4().hex[:10]}"

        cursor.execute(
            "INSERT INTO messages (id, session_id, sender, text, recalled_memories, saved_memories, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_msg_id, session_id, "user", payload.message, json.dumps([]), json.dumps([]), now_iso),
        )
        cursor.execute(
            "INSERT INTO messages (id, session_id, sender, text, recalled_memories, saved_memories, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ai_msg_id, session_id, "assistant", ai_reply, json.dumps(recalled_texts), json.dumps(saved_facts), now_iso),
        )

        conn.commit()
        conn.close()

        return ChatResponse(
            reply=ai_reply,
            session_id=session_id,
            session_title=session_title,
            recalled_memories=recalled_texts,
            saved_memories=saved_facts,
        )
    except Exception as e:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------
# SESSION & HISTORY ENDPOINTS
# ------------------------------------------------------------
@app.get("/api/sessions/{user_id}", response_model=List[SessionItem])
def get_user_sessions(user_id: str):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, user_id, title, is_pinned, created_at, updated_at FROM sessions WHERE user_id = ? ORDER BY is_pinned DESC, updated_at DESC",
            (user_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            SessionItem(
                id=r[0],
                user_id=r[1],
                title=r[2],
                is_pinned=bool(r[3]),
                created_at=r[4],
                updated_at=r[5],
            )
            for r in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sessions", response_model=SessionItem)
def create_new_session(payload: CreateSessionRequest):
    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        session_id = f"sess_{uuid.uuid4().hex[:10]}"
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (id, user_id, title, is_pinned, created_at, updated_at) VALUES (?, ?, ?, 0, ?, ?)",
            (session_id, payload.user_id, payload.title or "New Chat", now_iso, now_iso),
        )
        conn.commit()
        conn.close()

        return SessionItem(
            id=session_id,
            user_id=payload.user_id,
            title=payload.title or "New Chat",
            is_pinned=False,
            created_at=now_iso,
            updated_at=now_iso,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}/messages", response_model=List[MessageItem])
def get_session_messages(session_id: str):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, session_id, sender, text, recalled_memories, saved_memories, created_at FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            MessageItem(
                id=r[0],
                session_id=r[1],
                sender=r[2],
                text=r[3],
                recalled_memories=json.loads(r[4]) if r[4] else [],
                saved_memories=json.loads(r[5]) if r[5] else [],
                created_at=r[6],
            )
            for r in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/sessions/{session_id}/pin")
def toggle_pin(session_id: str, payload: UpdatePinRequest):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("UPDATE sessions SET is_pinned = ? WHERE id = ?", (1 if payload.is_pinned else 0, session_id))
        conn.commit()
        conn.close()
        return {"status": "success", "session_id": session_id, "is_pinned": payload.is_pinned}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/sessions/{session_id}/title")
def rename_session(session_id: str, payload: UpdateTitleRequest):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("UPDATE sessions SET title = ? WHERE id = ?", (payload.title, session_id))
        conn.commit()
        conn.close()
        return {"status": "success", "session_id": session_id, "title": payload.title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        conn.close()
        return {"status": "deleted", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------
# GLOBAL MEMORY ENDPOINTS
# ------------------------------------------------------------
@app.get("/api/memories/{user_id}", response_model=List[MemoryItem])
def get_all_memories(user_id: str):
    try:
        scroll_res, _ = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="user_id",
                        match=qmodels.MatchValue(value=user_id),
                    )
                ]
            ),
            limit=100,
        )
        return [
            MemoryItem(
                id=str(p.id),
                memory=str(p.payload.get("memory", "")),
                created_at=str(p.payload.get("created_at", "")) if p.payload.get("created_at") else None,
            )
            for p in scroll_res
            if p.payload and "memory" in p.payload
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/memories/{memory_id}")
def delete_memory(memory_id: str):
    try:
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=[memory_id],
        )
        return {"status": "deleted", "memory_id": memory_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "MemAI", "model": MODEL_ID}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)