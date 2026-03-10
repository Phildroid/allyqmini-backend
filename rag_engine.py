# rag_engine.py
# Extracted from AllyQ Mini notebook — do NOT run this in Jupyter, import it from api.py

import os
import time
import functools

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

import pypdf
from pptx import Presentation

# ── 1. ENVIRONMENT ──────────────────────────────────────────────────────────
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError("❌ GOOGLE_API_KEY not found. Add it to your .env file.")

device = "cpu"
print("✅ rag_engine loaded (cloud embeddings mode)")

# ── 2. TEXT SPLITTER ─────────────────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""]
)

# ── 3. LLM (Gemini) ──────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    convert_system_message_to_human=True,
    max_retries=2
)
print("✅ Gemini LLM ready")

# ── 4. EMBEDDINGS via direct REST call to v1 (bypasses all SDK v1beta routing) ──
import requests

class GoogleEmbeddings(Embeddings):
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.url = "https://generativelanguage.googleapis.com/v1/models/gemini-embedding-001:embedContent"

    def _embed(self, text):
        resp = requests.post(
            f"{self.url}?key={self.api_key}",
            json={
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": text}]}
            }
        )
        resp.raise_for_status()
        return resp.json()["embedding"]["values"]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

embeddings = GoogleEmbeddings()
print("✅ Google embeddings ready")
bge_embeddings = embeddings

# ── 5. GLOBAL VECTOR STORE ───────────────────────────────────────────────────
vector_store = None


# ── 6. EXTRACTION FUNCTIONS ──────────────────────────────────────────────────

def extract_from_pdf(file_path):
    docs = []
    reader = pypdf.PdfReader(file_path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        docs.append(Document(
            page_content=text,
            metadata={"page": i, "source": file_path}
        ))
    return docs


def extract_from_excel(file_path):
    sheets = pd.read_excel(file_path, sheet_name=None)
    docs = []
    for sheet_name, df in sheets.items():
        df = df.dropna(how="all").dropna(axis=1, how="all")
        headers = df.columns.tolist()
        for index, row in df.iterrows():
            row_parts = []
            for header in headers:
                val = row[header]
                if pd.notna(val):
                    row_parts.append(f"{header}: {val}")
            row_str = " | ".join(row_parts)
            docs.append(Document(
                page_content=f"SHEET: {sheet_name} | ROW {index}: {row_str}",
                metadata={
                    "source": file_path,
                    "sheet": sheet_name,
                    "row": index,
                    "is_tabular": True
                }
            ))
    return docs


def extract_from_pptx(file_path):
    prs = Presentation(file_path)
    docs = []
    for i, slide in enumerate(prs.slides):
        text = "\n".join([s.text for s in slide.shapes if hasattr(s, "text")])
        docs.append(Document(
            page_content=text,
            metadata={"slide": i, "source": file_path}
        ))
    return docs


# ── 7. INDEXING ──────────────────────────────────────────────────────────────

def process_and_index_file(file_path):
    global vector_store
    ext = os.path.splitext(file_path)[1].lower()

    print(f"🛠️  Indexing: {os.path.basename(file_path)}...")

    if ext == ".pdf":
        new_docs = extract_from_pdf(file_path)
        final_chunks = text_splitter.split_documents(new_docs)
    elif ext in [".xlsx", ".xls"]:
        final_chunks = extract_from_excel(file_path)
    elif ext == ".pptx":
        new_docs = extract_from_pptx(file_path)
        final_chunks = text_splitter.split_documents(new_docs)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if vector_store is not None:
        vector_store.add_documents(final_chunks)
        _cached_search.cache_clear()
        print(f"🚀 Added {len(final_chunks)} chunks to existing index.")
    else:
        print("🧠 Building fresh FAISS index...")
        vector_store = FAISS.from_documents(final_chunks, embeddings)

    vector_store.save_local("faiss_index")
    print(f"✅ Indexing complete — {len(final_chunks)} chunks saved.")
    return final_chunks


# ── 8. CACHED SEARCH ─────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=32)
def _cached_search(query: str, k: int):
    return vector_store.similarity_search(query, k=k)


# ── 9. QUERY FUNCTION ────────────────────────────────────────────────────────

def ask_documents(query: str, k: int = 10, return_docs: bool = False):
    global vector_store

    if vector_store is None:
        if os.path.exists("faiss_index"):
            print("💾 Loading FAISS index from disk...")
            vector_store = FAISS.load_local(
                "faiss_index", embeddings, allow_dangerous_deserialization=True
            )
        else:
            raise RuntimeError("No index found. Please upload and index a file first.")

    start = time.time()
    results = _cached_search(query, k)

    context_parts = []
    for res in results:
        source = os.path.basename(res.metadata.get("source", "Unknown"))
        loc = res.metadata.get(
            "sheet", res.metadata.get("slide", res.metadata.get("page", "Data"))
        )
        context_parts.append(f"\n--- FROM {source} ({loc}) ---\n{res.page_content}\n")

    context_text = "".join(context_parts)
    prompt = (
        "SYSTEM: You are a precision analyst. Answer based ONLY on the context below.\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION: {query}"
    )

    response = llm.invoke([HumanMessage(content=[{"type": "text", "text": prompt}])])
    answer = response.content

    print(f"⏱️  Latency: {time.time() - start:.2f}s")

    if return_docs:
        return answer, results
    return answer