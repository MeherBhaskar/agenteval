#!/usr/bin/env python3
"""
Clean Retriever Zoo Benchmark Runner
Imports retriever implementations and runs systematic evaluation.
"""

import os
import time
import json
import argparse
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from tqdm import tqdm

from retrievers import (
    BaseRetriever, BM25Retriever, DenseRetriever, SPLADERetriever, ColBERTRetriever,
    load_msmarco_sample, load_beir_dataset, RETRIEVER_REGISTRY, DATASET_REGISTRY
)


# ============================================================
# METRICS
# ============================================================

def ndcg_at_k(retrieved: List[str], relevant: Dict[str, int], k: int = 10) -> float:
    """Compute nDCG@k."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        rel = relevant.get(doc_id, 0)
        dcg += (2**rel - 1) / np.log2(i + 2)

    ideal_rels = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved: List[str], relevant: Dict[str, int], k: int = 100) -> float:
    """Compute Recall@k."""
    retrieved_set = set(retrieved[:k])
    relevant_set = set(relevant.keys())
    return len(retrieved_set & relevant_set) / len(relevant_set) if relevant_set else 0.0


def mrr_at_k(retrieved: List[str], relevant: Dict[str, int], k: int = 100) -> float:
    """Compute MRR@k."""
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_retriever(retriever: BaseRetriever, queries: Dict, qrels: Dict, k_values: List[int] = [10, 100]) -> Dict:
    """Evaluate a retriever on all metrics."""
    query_texts = list(queries.values())
    query_ids = list(queries.keys())

    print(f"  Running batch search on {len(query_texts)} queries...")
    start = time.time()
    all_results = retriever.batch_search(query_texts, k=max(k_values))
    total_latency = time.time() - start
    latency_ms = (total_latency / len(query_texts)) * 1000

    metrics = {"latency_ms": latency_ms}

    for k in k_values:
        ndcg_scores = []
        recall_scores = []
        mrr_scores = []

        for qid, results in zip(query_ids, all_results):
            retrieved_ids = [doc_id for doc_id, _ in results]
            relevant = qrels.get(qid, {})

            ndcg_scores.append(ndcg_at_k(retrieved_ids, relevant, k))
            recall_scores.append(recall_at_k(retrieved_ids, relevant, k))
            mrr_scores.append(mrr_at_k(retrieved_ids, relevant, k))

        metrics[f"ndcg@{k}"] = np.mean(ndcg_scores)
        metrics[f"recall@{k}"] = np.mean(recall_scores)
        metrics[f"mrr@{k}"] = np.mean(mrr_scores)

    return metrics


def estimate_cost(retriever_name: str, latency_ms: float) -> float:
    """Rough cost estimate per 1K queries in USD."""
    # GPU hour cost (A100 ~$2.50/hr)
    gpu_cost_per_hour = 2.50
    gpu_cost_per_ms = gpu_cost_per_hour / (3600 * 1000)

    if retriever_name == "bm25":
        return 0.02  # CPU only
    elif "bge" in retriever_name:
        return latency_ms * gpu_cost_per_ms * 1000
    elif "splade" in retriever_name:
        return latency_ms * gpu_cost_per_ms * 1000 * 1.5  # more compute
    elif "colbert" in retriever_name:
        return latency_ms * gpu_cost_per_ms * 1000 * 3  # most compute
    return 0.0


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Retriever Zoo Benchmark")
    parser.add_argument("--retrievers", nargs="+", default=["bm25", "bge-base", "bge-large", "splade", "colbert"])
    parser.add_argument("--datasets", nargs="+", default=["msmarco", "nfcorpus", "scifact"])
    parser.add_argument("--output", default="benchmark_results.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-docs", type=int, default=50000)
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--k", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    all_results = []

    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*60}")

        # Load dataset
        if dataset_name == "msmarco":
            data = load_msmarco_sample(max_docs=args.max_docs, max_queries=args.max_queries)
        else:
            data = DATASET_REGISTRY[dataset_name](max_queries=args.max_queries)

        corpus = data["corpus"]
        queries = data["queries"]
        qrels = data["qrels"]

        # Convert corpus to {doc_id: text} format
        corpus_texts = {}
        for doc_id, doc_info in corpus.items():
            if isinstance(doc_info, dict):
                text = doc_info.get("text", doc_info.get("body", ""))
            else:
                text = str(doc_info)
            corpus_texts[doc_id] = text

        print(f"Corpus: {len(corpus_texts)} docs | Queries: {len(queries)} | Qrels: {len(qrels)}")

        for retriever_name in args.retrievers:
            print(f"\n--- {retriever_name} ---")

            try:
                init_start = time.time()

                if retriever_name in ["bge-large", "bge-base"]:
                    model_name = "BAAI/bge-large-en-v1.5" if "large" in retriever_name else "BAAI/bge-base-en-v1.5"
                    retriever = DenseRetriever(model_name, corpus_texts, device=args.device)
                elif retriever_name == "bm25":
                    retriever = BM25Retriever(corpus_texts)
                elif retriever_name == "splade":
                    retriever = SPLADERetriever(corpus_texts, device=args.device)
                elif retriever_name == "colbert":
                    retriever = ColBERTRetriever(corpus_texts, device=args.device)
                else:
                    raise ValueError(f"Unknown retriever: {retriever_name}")

                init_time = time.time() - init_start
                print(f"  Init: {init_time:.1f}s")

                metrics = evaluate_retriever(retriever, queries, qrels, k_values=[10, 100])
                cost = estimate_cost(retriever_name, metrics["latency_ms"])

                result = {
                    "retriever": retriever_name,
                    "dataset": dataset_name,
                    "init_time_s": init_time,
                    "cost_per_1k_usd": cost,
                    **metrics
                }
                all_results.append(result)

                print(f"  nDCG@10: {metrics['ndcg@10']:.4f}")
                print(f"  Recall@100: {metrics['recall@100']:.4f}")
                print(f"  MRR@10: {metrics['mrr@10']:.4f}")
                print(f"  Latency: {metrics['latency_ms']:.1f}ms")
                print(f"  Cost/1K: ${cost:.4f}")

                # Cleanup
                del retriever
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  ERROR: {e}")
                all_results.append({
                    "retriever": retriever_name,
                    "dataset": dataset_name,
                    "error": str(e)
                })

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    json_path = args.output.replace(".csv", ".json")
    df.to_json(json_path, orient="records", indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS SAVED")
    print(f"{'='*60}")
    print(f"CSV: {args.output}")
    print(f"JSON: {json_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()