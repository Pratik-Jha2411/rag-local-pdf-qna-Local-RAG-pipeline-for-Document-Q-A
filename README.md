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
python ingest.py "C:\\Users\\prinx\\Downloads\\Acme Corp Employee Remote Work Policy.pdf" rag_index
```

2. Ask a question:

```bash
python qa.py rag_index "What is the policy for parental leave?"
```

Notes
- The default embedding model is `all-MiniLM-L6-v2` (local) and the default generator is `google/flan-t5-small` (small, instruction-tuned). Both run locally but may be slow on CPU.
- If you prefer a different local generation model, pass its HF model name to `qa.answer_question` or modify the `gen_model` parameter in `qa.py`.
- This implementation is intentionally simple (naive prompt + retrieval). For production use, add caching, more robust splitting, streaming, and better prompt engineering.
