import json
import os
from typing import List, Dict

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss


def read_pdf_text(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(text)
    return pages


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
    return chunks


def build_embeddings(docs: List[Dict], model_name: str = "all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def create_faiss_index(embeddings, index_path: str):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, index_path)


def ingest_pdf(pdf_path: str, out_dir: str, chunk_size: int = 500, overlap: int = 50):
    os.makedirs(out_dir, exist_ok=True)
    pages = read_pdf_text(pdf_path)

    docs = []
    for page_num, page_text in enumerate(pages, start=1):
        page_chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
        for i, chunk in enumerate(page_chunks):
            docs.append({"text": chunk, "meta": {"page": page_num, "chunk": i}})

    print(f"Prepared {len(docs)} chunks for embedding")

    embeddings = build_embeddings(docs)

    # Save metadata and texts
    meta_path = os.path.join(out_dir, "docs.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    # Create and save FAISS index
    index_path = os.path.join(out_dir, "index.faiss")
    create_faiss_index(embeddings, index_path)

    print("Ingestion complete. Index and metadata saved in:", out_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest a PDF and build a local FAISS vector store")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("out_dir", help="Output directory to store index and metadata")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    ingest_pdf(args.pdf, args.out_dir, chunk_size=args.chunk_size, overlap=args.overlap)
