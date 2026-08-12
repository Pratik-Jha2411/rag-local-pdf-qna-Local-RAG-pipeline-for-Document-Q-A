import json
import os
from typing import List

import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


def load_index_and_docs(out_dir: str):
    index_path = os.path.join(out_dir, "index.faiss")
    meta_path = os.path.join(out_dir, "docs.json")
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Index or metadata not found in out_dir")

    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return index, docs


def retrieve(query: str, index, docs, embed_model_name: str = "all-MiniLM-L6-v2", top_k: int = 3):
    embedder = SentenceTransformer(embed_model_name)
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = [docs[int(i)] for i in I[0] if i != -1]
    return results


def generate_answer(question: str, contexts: List[str], gen_model: str = "google/flan-t5-small") -> str:
    # concatenate contexts into prompt
    prompt = "Answer the question using only the provided context. If unknown, say you don't know.\n\n"
    prompt += "Context:\n" + "\n---\n".join(contexts) + "\n\n"
    prompt += "Question: " + question + "\nAnswer:"

    # Use a seq2seq model (Flan-T5 small by default).
    # Avoid transformers.pipeline here to keep compatibility; use model.generate.
    tokenizer = AutoTokenizer.from_pretrained(gen_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(gen_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

    gen_kwargs = {"max_new_tokens": 256, "do_sample": False}
    output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text.strip()


def answer_question(question: str, out_dir: str, embed_model: str = "all-MiniLM-L6-v2", gen_model: str = "google/flan-t5-small", top_k: int = 3):
    index, docs = load_index_and_docs(out_dir)
    results = retrieve(question, index, docs, embed_model_name=embed_model, top_k=top_k)
    contexts = [r["text"] for r in results]
    answer = generate_answer(question, contexts, gen_model=gen_model)
    return answer, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run simple local QA against FAISS index")
    parser.add_argument("out_dir", help="Directory where index and docs are stored")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    ans, ctx = answer_question(args.question, args.out_dir, top_k=args.top_k)
    print("ANSWER:\n", ans)
    print("\nRETRIEVED: ")
    for r in ctx:
        print(f"- Page {r['meta'].get('page')} chunk {r['meta'].get('chunk')}: {r['text'][:200].replace('\n',' ')}...")
