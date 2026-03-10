# api.py
# Local:  uvicorn api:app --reload --port 8000
# Render: uvicorn api:app --host 0.0.0.0 --port $PORT

import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import rag_engine

# ── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "frontend"))
HTML_FILE    = os.path.join(FRONTEND_DIR, "allyq-mini.html")
UPLOAD_DIR   = os.path.join(BASE_DIR, "knowledge_drop")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── APP SETUP ────────────────────────────────────────────────────────────────
app = FastAPI(title="AllyQ Mini API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SCHEMAS ──────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    k: int = 10

# ── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    # Serve the HTML file if running locally with frontend present
    if os.path.isfile(HTML_FILE):
        return FileResponse(HTML_FILE)
    # On Render (no frontend folder), just return a health check
    return JSONResponse({"status": "AllyQ Mini API is running"})


@app.get("/status")
def status():
    return {
        "engine": "ready",
        "index_loaded": rag_engine.vector_store is not None,
        "device": str(rag_engine.device),
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed_extensions = {".pdf", ".xlsx", ".xls", ".pptx"}
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: PDF, XLSX, XLS, PPTX"
        )

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"📥 Received: {file.filename}")

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
    
@app.get("/list-models")
def list_models():
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    models = [m.name for m in genai.list_models() if "embedContent" in m.supported_generation_methods]
    return {"embedding_models": models}