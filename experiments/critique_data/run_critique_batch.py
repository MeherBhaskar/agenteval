#!/usr/bin/env python3
"""
Batch critique generation for Paper 1 (Critique-Loop RAG).
Processes queries in batches, saves incrementally, handles rate limits.
"""

import json
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from rank_bm25 import BM25Okapi

# Configuration
LITELLM_BASE = "http://localhost:4000"
MODEL = "nemotron-ultra"
HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer not-used"}

CRITIQUE_PROMPT = """You are an expert evaluator of retrieval-augmented generation (RAG) systems.
Given a user query, retrieved documents, and a generated answer, provide a structured critique.

QUERY: {query}

RETRIEVED DOCUMENTS:
{documents}

GENERATED ANSWER:
{answer}

Evaluate and output JSON with exactly these fields:
{{
  "missing_info": ["specific fact 1", "specific fact 2"],  // Facts needed but not in docs
  "irrelevant_docs": ["doc_id_1", "doc_id_2"],  // Doc IDs that added noise
  "hallucination_spans": ["exact phrase from answer not in docs"],  // Unverified claims
  "quality_score": 1-5,  // 1=unusable, 2=poor, 3=ok, 4=good, 5=excellent
  "reasoning": "Brief explanation of score"
}}

Be precise. "Missing info" must be specific facts the answer should have included.
"Irrelevant docs" must be doc IDs from the retrieved list.
"Hallucination spans" must be exact substrings from the answer."""

def generate(prompt, max_tokens=1024, temperature=0.3):
    payload = {
        "model": "nemotron-ultra",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95
    }
    r = requests.post(
        "http://localhost:4000/v1/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": "Bearer not-used"},
        json=payload, timeout=180
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def load_queries(path, limit):
    queries = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit: break
            q = json.loads(line)
            queries.append({"id": q["_id"], "text": q["text"]})
    return queries

def load_corpus(path, limit):
    docs = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit: break
            d = json.loads(line)
            docs[d["_id"]] = d.get("text", "")
    return docs

def build_bm25(corpus):
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did] for did in doc_ids]
    return BM25Okapi([t.lower().split() for t in doc_texts]), doc_ids, doc_texts

def process_query(qid, query, bm25, doc_ids, doc_texts, corpus):
    # Retrieve
    scores = bm25.get_scores(query.lower().split())
    top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
    docs = [{"id": doc_ids[i], "text": doc_texts[i], "score": float(scores[i])} for i in top_k]

    # Generate answer
    doc_text = "\n\n".join([f"[{d['id']}] {d['text']}" for d in docs])
    prompt = f"""Answer the question using ONLY the provided documents. If the answer isn't in the docs, say "I don't know based on the provided documents."

QUESTION: {query}

DOCUMENTS:
{doc_text}

ANSWER:"""

    answer = generate(prompt, max_tokens=256, temperature=0.3).strip()

    # Generate critique
    doc_text = "\n\n".join([f"[{d['id']}] {d['text']}" for d in docs])
    critique_prompt = f"""You are an expert evaluator of retrieval-augmented generation (RAG) systems.
Given a user query, retrieved documents, and a generated answer, provide a structured critique.

QUERY: {query}

RETRIEVED DOCUMENTS:
{doc_text}

GENERATED ANSWER:
{answer}

Evaluate and output JSON with exactly these fields:
{{
  "missing_info": ["specific fact 1", "specific fact 2"],
  "irrelevant_docs": ["doc_id_1", "doc_id_2"],
  "hallucination_spans": ["exact phrase from answer not in docs"],
  "quality_score": 1-5,
  "reasoning": "Brief explanation of score"
}}

Be precise. "Missing info" must be specific facts the answer should have included.
"Irrelevant docs" must be doc IDs from the retrieved list.
"Hallucination spans" must be exact substrings from the answer."""

    critique_text = generate(critique_prompt, max_tokens=1024, temperature=0.1)

    # Parse critique JSON
    try:
        start = critique_text.find("{")
        end = critique_text.rfind("}") + 1
        critique = json.loads(critique_text[start:end])
    except:
        critique = {"missing_info": [], "irrelevant_docs": [], "hallucination_spans": [], "quality_score": 3, "reasoning": "Parse error"}

    return {
        "query": query, "query_id": qid,
        "retrieved_docs": [{"id": d["id"], "text": d["text"], "score": d["score"]} for d in docs],
        "generated_answer": answer,
        "critique": critique,
        "retriever_used": "bm25",
        "generator_used": "nemotron-ultra"
    }

def main():
    import sys
    os.chdir("/home/meher/agentic-ai-research/experiments/critique_data")

    # Config
    MAX_QUERIES = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    OUTPUT = f"critique_train_{MAX_QUERIES}.jsonl"
    N_WORKERS = 2

    # Load data
    print(f"Loading {MAX_QUERIES} queries from MS MARCO...")
    queries = load_queries("datasets/msmarco/queries.jsonl", MAX_QUERIES)
    corpus = load_corpus("datasets/msmarco/corpus.jsonl", 5000)

    # Build BM25
    print("Building BM25 index...")
    bm25, doc_ids, doc_texts = build_bm25(corpus)

    # Process in parallel
    print(f"Processing {len(queries)} queries with {N_WORKERS} workers...")
    os.makedirs("output", exist_ok=True)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(process_query, q["id"], q["text"], bm25, doc_ids, doc_texts, corpus): q
            for q in queries
        }

        with open(f"output/{OUTPUT}", "w") as f:
            for future in tqdm(as_completed(futures), total=len(futures)):
                q = futures[future]
                try:
                    result = future.result()
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                except Exception as e:
                    print(f"Error on {q['id']}: {e}")

    print(f"Done! Saved to output/{OUTPUT}")

if __name__ == "__main__":
    main()