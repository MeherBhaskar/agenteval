#!/usr/bin/env python3
"""
Retriever Zoo Benchmark - Lightweight Local Evaluation
Runs 5 retrievers on 3 datasets, outputs CSV for Papers 1, 2, 4.
No distributed training, single GPU, sequential execution.
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
from tqdm import tqdm

# Retriever implementations (local, no external services)
try:
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    import faiss
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install: pip install sentence-transformers rank-bm25 faiss-cpu tqdm")
    exit(1)


# ============================================================
# DATASETS
# ============================================================

def load_beir_dataset(name: str, split: str = "test", max_queries: int = 1000) -> Dict:
    """Load BEIR dataset (downloads on first run)."""
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    data_path = util.download_and_unzip(url, "datasets")

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    # Limit queries for speed
    query_ids = list(queries.keys())[:max_queries]
    queries = {qid: queries[qid] for qid in query_ids}
    qrels = {qid: qrels[qid] for qid in query_ids if qid in qrels}

    return {"corpus": corpus, "queries": queries, "qrels": qrels}


def load_msmarco_sample(max_docs: int = 50000, max_queries: int = 1000) -> Dict:
    """Load MS MARCO passage ranking sample."""
    # Using BEIR's MS MARCO
    return load_beir_dataset("msmarco", "dev", max_queries)


# ============================================================
# RETRIEVERS
# ============================================================

class BM25Retriever:
    def __init__(self, corpus: Dict[str, Dict]):
        self.corpus = corpus
        self.doc_ids = list(corpus.keys())
        self.doc_texts = [corpus[did].get("text", "") for did in self.doc_ids]
        tokenized = [text.split() for text in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 100) -> List[tuple]:
        scores = self.bm25.get_scores(query.split())
        top_k = np.argpartition(scores, -k)[-k:]
        top_k = top_k[np.argsort(scores[top_k])][::-1]
        return [(self.doc_ids[i], float(scores[i])) for i in top_k]

    def batch_search(self, queries: List[str], k: int = 100) -> List[List[tuple]]:
        return [self.search(q, k) for q in queries]


class DenseRetriever:
    def __init__(self, model_name: str, corpus: Dict[str, Dict], device: str = "cuda"):
        self.model = SentenceTransformer(model_name, device=device)
        self.corpus = corpus
        self.doc_ids = list(corpus.keys())
        self.doc_texts = [corpus[did].get("text", "") for did in self.doc_ids]
        self.device = device
        self._build_index()

    def _build_index(self):
        print(f"Encoding {len(self.doc_texts)} documents...")
        batch_size = 256
        embeddings = []
        for i in tqdm(range(0, len(self.doc_texts), batch_size)):
            batch = self.doc_texts[i:i+batch_size]
            emb = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings.append(emb)
        self.doc_embeddings = np.vstack(embeddings).astype(np.float32)

        # FAISS index
        dim = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(self.doc_embeddings)
        self.index.add(self.doc_embeddings)
        print(f"Index built: {self.index.ntotal} vectors")

    def search(self, query: str, k: int = 100) -> List[tuple]:
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb.astype(np.float32), k)
        return [(self.doc_ids[idx], float(score)) for idx, score in zip(indices[0], scores[0])]

    def batch_search(self, queries: List[str], k: int = 100) -> List[List[tuple]]:
        q_embs = self.model.encode(queries, convert_to_numpy=True)
        faiss.normalize_L2(q_embs)
        scores, indices = self.index.search(q_embs.astype(np.float32), k)
        results = []
        for i in range(len(queries)):
            results.append([(self.doc_ids[idx], float(score)) for idx, score in zip(indices[i], scores[i])])
        return results


class SPLADERetriever:
    """SPLADE via sentence-transformers (splade-cocondenser-ensembledistil)"""
    def __init__(self, corpus: Dict[str, Dict], device: str = "cuda"):
        # Using a sparse encoder approximation
        self.model = SentenceTransformer("naver/splade-cocondenser-ensembledistil", device=device)
        self.corpus = corpus
        self.doc_ids = list(corpus.keys())
        self.doc_texts = [corpus[did].get("text", "") for did in self.doc_ids]
        self.device = device
        self._build_index()

    def _build_index(self):
        print(f"Encoding {len(self.doc_texts)} documents with SPLADE...")
        batch_size = 128  # Smaller for SPLADE
        embeddings = []
        for i in tqdm(range(0, len(self.doc_texts), batch_size)):
            batch = self.doc_texts[i:i+batch_size]
            emb = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings.append(emb)
        self.doc_embeddings = np.vstack(embeddings).astype(np.float32)

        dim = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(self.doc_embeddings)
        self.index.add(self.doc_embeddings)
        print(f"SPLADE index built: {self.index.ntotal} vectors")

    def search(self, query: str, k: int = 100) -> List[tuple]:
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb.astype(np.float32), k)
        return [(self.doc_ids[idx], float(score)) for idx, score in zip(indices[0], scores[0])]

    def batch_search(self, queries: List[str], k: int = 100) -> List[List[tuple]]:
        q_embs = self.model.encode(queries, convert_to_numpy=True)
        faiss.normalize_L2(q_embs)
        scores, indices = self.index.search(q_embs.astype(np.float32), k)
        results = []
        for i in range(len(queries)):
            results.append([(self.doc_ids[idx], float(score)) for idx, score in zip(indices[i], scores[i])])
        return results


class ColBERTRetriever:
    """ColBERT via sentence-transformers (colbert-ir/colbertv2.0)"""
    def __init__(self, corpus: Dict[str, Dict], device: str = "cuda"):
        # Note: Full ColBERT needs PLAID index; using bi-encoder approx for benchmark
        self.model = SentenceTransformer("colbert-ir/colbertv2.0", device=device)
        self.corpus = corpus
        self.doc_ids = list(corpus.keys())
        self.doc_texts = [corpus[did].get("text", "") for did in self.doc_ids]
        self.device = device
        self._build_index()

    def _build_index(self):
        print(f"Encoding {len(self.doc_texts)} documents with ColBERT...")
        batch_size = 64  # ColBERT larger
        embeddings = []
        for i in tqdm(range(0, len(self.doc_texts), batch_size)):
            batch = self.doc_texts[i:i+batch_size]
            emb = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings.append(emb)
        self.doc_embeddings = np.vstack(embeddings).astype(np.float32)

        dim = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(self.doc_embeddings)
        self.index.add(self.doc_embeddings)
        print(f"ColBERT index built: {self.index.ntotal} vectors")

    def search(self, query: str, k: int = 100) -> List[tuple]:
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb.astype(np.float32), k)
        return [(self.doc_ids[idx], float(score)) for idx, score in zip(indices[0], scores[0])]

    def batch_search(self, queries: List[str], k: int = 100) -> List[List[tuple]]:
        q_embs = self.model.encode(queries, convert_to_numpy=True)
        faiss.normalize_L2(q_embs)
        scores, indices = self.index.search(q_embs.astype(np.float32), k)
        results = []
        for i in range(len(queries)):
            results.append([(self.doc_ids[idx], float(score)) for idx, score in zip(indices[i], scores[i])])
        return results


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


def evaluate_retriever(retriever, queries: Dict, qrels: Dict, k_values: List[int] = [10, 100]) -> Dict:
    """Evaluate a retriever on all metrics."""
    query_texts = list(queries.values())
    query_ids = list(queries.keys())

    print(f"Running batch search on {len(query_texts)} queries...")
    start = time.time()
    all_results = retriever.batch_search(query_texts, k=max(k_values))
    latency = (time.time() - start) / len(query_texts) * 1000  # ms per query

    metrics = {f"latency_ms": latency}

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


# ============================================================
# MAIN
# ============================================================

RETRIEVERS_CONFIG = {
    "bm25": {"class": BM25Retriever, "model": None},
    "bge-large": {"class": DenseRetriever, "model": "BAAI/bge-large-en-v1.5"},
    "bge-base": {"class": DenseRetriever, "model": "BAAI/bge-base-en-v1.5"},
    "splade": {"class": SPLADERetriever, "model": None},
    "colbert": {"class": ColBERTRetriever, "model": None},
}

DATASETS_CONFIG = {
    "msmarco": {"loader": load_msmarco_sample, "args": {"max_docs": 50000, "max_queries": 1000}},
    "nfcorpus": {"loader": load_beir_dataset, "args": {"name": "nfcorpus", "max_queries": 500}},
    "scifact": {"loader": load_beir_dataset, "args": {"name": "scifact", "max_queries": 500}},
}


def main():
    parser = argparse.ArgumentParser(description="Retriever Zoo Benchmark")
    parser.add_argument("--retrievers", nargs="+", default=list(RETRIEVERS_CONFIG.keys()))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS_CONFIG.keys()))
    parser.add_argument("--output", default="experiments/retriever_zoo/benchmark_results.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k", type=int, default=100, help="Max k for retrieval")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    all_results = []

    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Loading dataset: {dataset_name}")
        print(f"{'='*60}")

        dataset_config = DATASETS_CONFIG[dataset_name]
        data = dataset_config["loader"](**dataset_config["args"])
        corpus, queries, qrels = data["corpus"], data["queries"], data["qrels"]

        print(f"Corpus: {len(corpus)} docs | Queries: {len(queries)} | Qrels: {len(qrels)}")

        for retriever_name in args.retrievers:
            print(f"\n--- {retriever_name} on {dataset_name} ---")

            config = RETRIEVERS_CONFIG[retriever_name]
            retriever_class = config["class"]
            model_name = config["model"]

            try:
                # Initialize retriever
                init_start = time.time()
                if model_name:
                    retriever = retriever_class(model_name, corpus, device=args.device)
                else:
                    retriever = retriever_class(corpus)
                init_time = time.time() - init_start
                print(f"Init time: {init_time:.1f}s")

                # Evaluate
                metrics = evaluate_retriever(retriever, queries, qrels, k_values=[10, 100])

                # Estimate cost (rough)
                if "bge" in retriever_name or "splade" in retriever_name or "colbert" in retriever_name:
                    # GPU inference cost estimate
                    cost_per_1k = metrics["latency_ms"] / 1000 * 2.50  # $2.50/hr A100
                else:
                    cost_per_1k = 0.02  # BM25 on CPU

                result = {
                    "retriever": retriever_name,
                    "dataset": dataset_name,
                    "init_time_s": init_time,
                    "cost_per_1k_usd": cost_per_1k,
                    **metrics
                }
                all_results.append(result)

                print(f"  nDCG@10: {metrics['ndcg@10']:.4f}")
                print(f"  Recall@100: {metrics['recall@100']:.4f}")
                print(f"  Latency: {metrics['latency_ms']:.1f}ms")
                print(f"  Est. cost/1K: ${cost_per_1k:.4f}")

                # Clear GPU memory
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
    import pandas as pd
    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"\n✅ Results saved to {args.output}")
    print(df.to_string())

    # Also save as JSON for paper tables
    json_path = args.output.replace(".csv", ".json")
    df.to_json(json_path, orient="records", indent=2)
    print(f"✅ JSON saved to {json_path}")


if __name__ == "__main__":
    main()