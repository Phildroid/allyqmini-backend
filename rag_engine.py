# rag_engine.py
# Extracted from AllyQ Mini notebook — do NOT run this in Jupyter, import it from api.py

import os
import io
import base64
import time
import functools

import torch
import numpy as np
import pandas as pd
from PIL import Image
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import fitz  # PyMuPDF
from pptx import Presentation

# ── 1. ENVIRONMENT ──────────────────────────────────────────────────────────
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError("❌ GOOGLE_API_KEY not found. Add it to your .env file.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ rag_engine loaded on {device.upper()}")

# ── 2. TEXT SPLITTER ─────────────────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""]
)

# ── 3. LLM (Gemini) ──────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",          # swap to "gemini-3-flash-preview" if you have access
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    convert_system_message_to_human=True,
    max_retries=2
)
print("✅ Gemini LLM ready")

# ── 4. BGE EMBEDDINGS ────────────────────────────────────────────────────────
bge_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)
print("✅ BGE embeddings ready")

# ── 5. GLOBAL VECTOR STORE ───────────────────────────────────────────────────
# This stays in memory so /upload and /query share the same index
vector_store = None


# ── 6. EXTRACTION FUNCTIONS ──────────────────────────────────────────────────

def extract_from_pdf(file_path, output_dir):
    docs = []
    pdf = fitz.open(file_path)
    for i in range(len(pdf)):
        page = pdf[i]
        text = page.get_text("text")
        img_paths = []
        for img_idx, img in enumerate(page.get_images(full=True)):
            base_img = pdf.extract_image(img[0])
            path = os.path.join(output_dir, f"pdf_{i}_{img_idx}.{base_img['ext']}")
            with open(path, "wb") as f:
                f.write(base_img["image"])
            img_paths.append(path)
        docs.append(Document(
            page_content=text,
            metadata={"page": i, "images": img_paths, "source": file_path}
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


def extract_from_pptx(file_path, output_dir):
    prs = Presentation(file_path)
    docs = []
    for i, slide in enumerate(prs.slides):
        text = "\n".join([s.text for s in slide.shapes if hasattr(s, "text")])
        img_paths = []
        for j, shape in enumerate(slide.shapes):
            if hasattr(shape, "image"):
                img_ext = shape.image.ext
                path = os.path.join(output_dir, f"ppt_{i}_{j}.{img_ext}")
                with open(path, "wb") as f:
                    f.write(shape.image.blob)
                img_paths.append(path)
        docs.append(Document(
            page_content=text,
            metadata={"slide": i, "images": img_paths, "source": file_path}
        ))
    return docs


# ── 7. INDEXING ──────────────────────────────────────────────────────────────

def process_and_index_file(file_path):
    global vector_store
    image_output_dir = "extracted_images"
    os.makedirs(image_output_dir, exist_ok=True)
    ext = os.path.splitext(file_path)[1].lower()

    print(f"🛠️  Indexing: {os.path.basename(file_path)}...")

    if ext == ".pdf":
        new_docs = extract_from_pdf(file_path, image_output_dir)
        final_chunks = text_splitter.split_documents(new_docs)
    elif ext in [".xlsx", ".xls"]:
        final_chunks = extract_from_excel(file_path)
    elif ext == ".pptx":
        new_docs = extract_from_pptx(file_path, image_output_dir)
        final_chunks = text_splitter.split_documents(new_docs)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if vector_store is not None:
        vector_store.add_documents(final_chunks)
        _cached_search.cache_clear()   # invalidate cache so new data is searchable
        print(f"🚀 Added {len(final_chunks)} chunks to existing index.")
    else:
        print("🧠 Building fresh FAISS index...")
        vector_store = FAISS.from_documents(final_chunks, bge_embeddings)

    vector_store.save_local("faiss_index")
    print(f"✅ Indexing complete — {len(final_chunks)} chunks saved.")
    return final_chunks


# ── 8. CACHED SEARCH ─────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=32)
def _cached_search(query: str, k: int):
    return vector_store.similarity_search(query, k=k)


# ── 9. HELPER: IMAGE → BASE64 ────────────────────────────────────────────────

def _get_image_b64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── 10. QUERY FUNCTION ───────────────────────────────────────────────────────

def ask_documents(query: str, k: int = 10, max_images: int = 2, return_docs: bool = False):
    """
    Query the indexed documents.
    Returns answer string, or (answer, source_docs) if return_docs=True.
    """
    global vector_store

    if vector_store is None:
        # Try loading from disk if it exists
        if os.path.exists("faiss_index"):
            print("💾 Loading FAISS index from disk...")
            vector_store = FAISS.load_local(
                "faiss_index", bge_embeddings, allow_dangerous_deserialization=True
            )
        else:
            raise RuntimeError("No index found. Please upload and index a file first.")

    start = time.time()
    results = _cached_search(query, k)

    context_parts = []
    images_to_send = []

    for res in results:
        source = os.path.basename(res.metadata.get("source", "Unknown"))
        loc = res.metadata.get(
            "sheet", res.metadata.get("slide", res.metadata.get("page", "Data"))
        )
        context_parts.append(f"\n--- FROM {source} ({loc}) ---\n{res.page_content}\n")
        if "images" in res.metadata:
            images_to_send.extend(res.metadata["images"])

    context_text = "".join(context_parts)

    prompt = (
        "SYSTEM: You are a precision analyst. Answer based ONLY on the context below.\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION: {query}"
    )

    content = [{"type": "text", "text": prompt}]

    unique_images = list(dict.fromkeys(images_to_send))[:max_images]
    for img_path in unique_images:
        try:
            b64 = _get_image_b64(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        except Exception as e:
            print(f"⚠️  Could not load image {img_path}: {e}")

    response = llm.invoke([HumanMessage(content=content)])
    answer = response.content

    print(f"⏱️  Latency: {time.time() - start:.2f}s")

    if return_docs:
        return answer, results
    return answer