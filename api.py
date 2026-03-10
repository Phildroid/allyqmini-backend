# api.py
# Run with: uvicorn api:app --reload --port 8000

import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import rag_engine  # your extracted notebook logic

# ── PATHS ────────────────────────────────────────────────────────────────────
# Resolve paths relative to THIS file, not the working directory
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))          # .../allyq-mini/backend
FRONTEND_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "frontend"))
HTML_FILE    = os.path.join(FRONTEND_DIR, "allyq-mini.html")

# ── APP SETUP ────────────────────────────────────────────────────────────────
app = FastAPI(title="AllyQ Mini API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SERVE FRONTEND ───────────────────────────────────────────────────────────
if os.path.isdir(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
    print(f"✅ Serving frontend from: {FRONTEND_DIR}")
else:
    print(f"⚠️  Frontend folder not found at: {FRONTEND_DIR}")

@app.get("/")
def root():
    if os.path.isfile(HTML_FILE):
        return FileResponse(HTML_FILE)
    return {
        "message": "AllyQ Mini API is running. "
                   "Frontend not found — open allyq-mini.html directly in your browser."
    }


# ── SCHEMAS ──────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    k: int = 10


# ── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/status")
def status():
    """Health check — frontend polls this on load to show engine status."""
    return {
        "engine": "ready",
        "index_loaded": rag_engine.vector_store is not None,
        "device": rag_engine.device,
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Receives a file from the frontend, saves it to knowledge_drop/,
    and indexes it into FAISS via rag_engine.
    """
    allowed_extensions = {".pdf", ".xlsx", ".xls", ".pptx"}
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: PDF, XLSX, XLS, PPTX"
        )

    # Save uploaded file next to api.py in backend/knowledge_drop/
    knowledge_dir = os.path.join(BASE_DIR, "knowledge_drop")
    os.makedirs(knowledge_dir, exist_ok=True)
    save_path = os.path.join(knowledge_dir, file.filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"📥 Received: {file.filename}")

    # Index it
    try:
        chunks = rag_engine.process_and_index_file(save_path)
        return {
            "success": True,
            "filename": file.filename,
            "chunks": len(chunks),
            "message": f"Successfully indexed {len(chunks)} chunks"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.post("/query")
def query_documents(req: QueryRequest):
    """
    Receives a query from the frontend chat input,
    runs RAG retrieval + Gemini reasoning, returns answer + sources.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        answer, source_docs = rag_engine.ask_documents(
            query=req.query,
            k=req.k,
            return_docs=True
        )

        sources = []
        for doc in source_docs:
            sources.append({
                "file": os.path.basename(doc.metadata.get("source", "Unknown")),
                "loc": doc.metadata.get(
                    "sheet",
                    doc.metadata.get(
                        "slide",
                        f"Page {doc.metadata.get('page', '?')}"
                    )
                )
            })

        return {
            "success": True,
            "answer": answer,
            "sources": sources,
            "chunks_searched": req.k
        }

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")