# api.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, shutil, uuid

from rag_engine import process_and_index_file, ask_documents, clear_session, list_sessions

UPLOAD_DIR = "knowledge_drop"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="AllyQ Mini API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MODELS ────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    session_id: str
    k: int = 10

class ClearRequest(BaseModel):
    session_id: str

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "AllyQ Mini API"}

@app.get("/status")
def status():
    return {
        "status": "ok",
        "device": "cpu",
        "active_sessions": len(list_sessions()),
        "session_ids": list_sessions()
    }

@app.post("/upload")
async def upload(file: UploadFile = File(...), session_id: str = "default"):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".xlsx", ".xls", ".pptx"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Store each session's files in its own subfolder to avoid collisions
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    dest = os.path.join(session_dir, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks = process_and_index_file(dest, session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "indexed", "file": file.filename, "chunks": chunks, "session_id": session_id}

@app.post("/query")
async def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        answer, sources = ask_documents(req.query, req.session_id, req.k)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"answer": answer, "sources": sources, "session_id": req.session_id}

@app.post("/clear-session")
async def clear(req: ClearRequest):
    clear_session(req.session_id)
    return {"status": "cleared", "session_id": req.session_id}