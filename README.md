# rag-local-pdf-qna

A minimal, fully-local Retrieval-Augmented Generation (RAG) pipeline for doing Q&A over a single PDF.

This project ingests a PDF, creates a FAISS vector index of sentence-transformer embeddings, and runs a local seq2seq model (via `transformers`) to generate answers from retrieved context. It is designed to run offline on your machine.

---

## Quickstart

1. Create a Python environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r rag_local/requirements.txt
```

3. Ingest a PDF to build the index and metadata. Example (replace with your PDF path):

```powershell
python rag_local/ingest.py "C:\Users\prinx\Downloads\Acme Corp Employee Remote Work Policy.pdf" rag_local/rag_index
```

This produces `rag_local/rag_index/docs.json` and `rag_local/rag_index/index.faiss`.

4. Ask a question against the ingested document:

```powershell
python rag_local/qa.py rag_local/rag_index "According to the policy, what are the eligibility rules for parental leave?"
```

---

## Files

- `rag_local/ingest.py` — parses a PDF, chunks text, computes embeddings (`sentence-transformers`), builds and saves a FAISS index and `docs.json` metadata.
- `rag_local/qa.py` — loads the FAISS index and metadata, retrieves top-k chunks for a question, and uses a local `transformers` seq2seq model to generate a final answer.
- `rag_local/requirements.txt` — Python dependencies used for the project.

---

## Models and Performance

- Embeddings: `sentence-transformers` default (`all-MiniLM-L6-v2`) — fast and compact.
- Generator: default `google/flan-t5-small` in `qa.py`. This runs on CPU (or GPU if available). On CPU it can be slow; consider a smaller model or a quantized variant if you need speed.

If you have a CUDA GPU and torch installed with CUDA support, `qa.py` will run the model on GPU automatically.

---

## Configuration and Tips

- Chunking: The ingestion script uses a simple chunking strategy. If answers omit details, reduce chunk size or increase overlap in `ingest.py`.
- Retrieval: Increase `k` in `qa.py` retrieval to include more context if answers appear incomplete.
- Memory / Speed: For large PDFs or many documents, switch to a disk-backed index or quantized models.

---

## Troubleshooting

- KeyError about `text2text-generation` pipeline: `qa.py` avoids `pipeline(...)` and calls `model.generate()` directly to remain compatible across `transformers` versions.
- If you see CUDA / GPU errors, confirm `torch` was installed with CUDA support and that a compatible GPU is present. Otherwise force CPU with `TORCH_DEVICE=cpu` environment variable.

---

## Example commands (copy-paste)

```powershell
# Install deps
python -m pip install -r rag_local/requirements.txt

# Ingest
python rag_local/ingest.py "C:\path\to\your.pdf" rag_local/rag_index

# QA
python rag_local/qa.py rag_local/rag_index "Give me the main headings and sections from the document."
```

---

## Contributing

Feel free to open issues or submit PRs for:

- smarter chunking logic
- switching embedding / generation models
- quantization helpers and performance scripts

---

## License

MIT
# Local Naive Document QnA (RAG-lite)

This project provides a minimal local RAG-style QnA pipeline:

- `ingest.py` — load a PDF, chunk it, compute embeddings with `sentence-transformers`, build a FAISS index, and save metadata.
- `qa.py` — load the FAISS index, retrieve top-k chunks for a query, and generate an answer using a local transformers model (default: `google/flan-t5-small`).

Requirements
```
pip install -r requirements.txt
```

Basic usage

1. Ingest your PDF (replace the path and out directory):

```bash
python ingest.py "FILE_PATH" rag_index
```

2. Ask a question:

```bash
python qa.py rag_index "What is the policy for parental leave?"
```

Notes
- The default embedding model is `all-MiniLM-L6-v2` (local) and the default generator is `google/flan-t5-small` (small, instruction-tuned). Both run locally but may be slow on CPU.
- If you prefer a different local generation model, pass its HF model name to `qa.answer_question` or modify the `gen_model` parameter in `qa.py`.
- This implementation is intentionally simple (naive prompt + retrieval). For production use, add caching, more robust splitting, streaming, and better prompt engineering.
