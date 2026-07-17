"""
Retriever implementations for the Retriever Zoo benchmark.
Each retriever implements a common interface:
- __init__(model_or_corpus, corpus=None, device="cuda")
- batch_search(queries: List[str], k: int) -> List[List[Tuple[str, float]]]
"""

import time
import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss


class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""

    @abstractmethod
    def batch_search(self, queries: List[str], k: int) -> List[List[Tuple[str, float]]]:
        """Return list of (doc_id, score) pairs for each query."""
        pass


class BM25Retriever(BaseRetriever):
    """BM25 retriever using rank-bm25."""

    def __init__(self, corpus: Dict[str, str]):
        self.corpus = corpus
        self.doc_ids = list(corpus.keys())
        self.doc_texts = [corpus[did] for did in self.doc_ids]

        # Tokenize for BM25
        print("Tokenizing corpus for BM25...")
        tokenized = [text.lower().split() for text in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized)
        print(f"BM25 index built for {len(self.doc_ids)} documents")

    def batch_search(self, queries: List[str], k: int) -> List[List[Tuple[str, float]]]:
        all_results = []
        for query in queries:
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            top_k_idx = np.argpartition(scores, -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])][::-1]
            results = [(self.doc_ids[i], float(scores[i])) for i in top_k_idx]
            all_results.append(results)
        return all_results


class DenseRetriever(BaseRetriever):
    """Dense retriever using SentenceTransformers + FAISS."""

    def __init__(self, model_name: str, corpus: Dict[str, str], device: str = "cuda"):
        self.model_name = model_name
        self.corpus = corpus
        self.device = device

        print(f"Loading {model_name} on {device}...")
        self.model = SentenceTransformer(model_name, device=device)

        self.doc_ids = list(corpus.keys())
        self.doc_texts = [corpus[did] for did in self.doc_ids]

        # Build FAISS index
        print("Encoding corpus...")
        batch_size = 256
        embeddings = []
        for i in range(0, len(self.doc_texts), batch_size):
            batch = self.doc_texts[i:i+batch_size]
            emb = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False, device=device)
            embeddings.append(emb)
        self.doc_embeddings = np.vstack(embeddings).astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(self.doc_embeddings)

        # Build index
        dim = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.doc_embeddings)

        print(f"FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    def batch_search(self, queries: List[str], k: int) -> List[List[Tuple[str, float]]]:
        # Encode queries
        query_embeddings = self.model.encode(queries, convert_to_numpy=True, show_progress_bar=False, device=self.device)
        query_embeddings = query_embeddings.astype(np.float32)
        faiss.normalize_L2(query_embeddings)

        # Search
        scores, indices = self.index.search(query_embeddings, k)

        # Format results
        all_results = []
        for i in range(len(queries)):
            results = []
            for j in range(k):
                idx = indices[i][j]
                if idx >= 0:
                    doc_id = self.doc_ids[idx]
                    score = float(scores[i][j])
                    results.append((doc_id, score))
            all_results.append(results)
        return all_results


class SPLADERetriever(BaseRetriever):
    """SPLADE retriever (sparse learned embeddings)."""

    def __init__(self, corpus: Dict[str, str], device: str = "cuda"):
        # Using splade model from sentence-transformers
        self.model = SentenceTransformer("naver/splade-v3-distilbert", device=device)
        self.corpus = corpus
        self.device = device

        self.doc_ids = list(corpus.keys())
        self.doc_texts = [corpus[did] for did in self.doc_ids]

        # SPLADE outputs sparse vectors - use dot product via FAISS
        print("Encoding corpus with SPLADE...")
        batch_size = 128
        embeddings = []
        for i in range(0, len(self.doc_texts), batch_size):
            batch = self.doc_texts[i:i+batch_size]
            emb = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False, device=device)
            embeddings.append(emb)
        self.doc_embeddings = np.vstack(embeddings).astype(np.float32)

        # SPLADE uses dot product (not normalized)
        dim = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.doc_embeddings)
        print(f"SPLADE index built: {self.index.ntotal} vectors")

    def batch_search(self, queries: List[str], k: int) -> List[List[Tuple[str, float]]]:
        query_embeddings = self.model.encode(queries, convert_to_numpy=True, show_progress_bar=False, device=self.device)
        query_embeddings = query_embeddings.astype(np.float32)

        scores, indices = self.index.search(query_embeddings, k)

        all_results = []
        for i in range(len(queries)):
            results = []
            for j in range(k):
                idx = indices[i][j]
                if idx >= 0:
                    doc_id = self.doc_ids[idx]
                    score = float(scores[i][j])
                    results.append((doc_id, score))
            all_results.append(results)
        return all_results


class ColBERTRetriever(BaseRetriever):
    """ColBERT retriever (late interaction). Uses simplified version for benchmarking."""

    def __init__(self, corpus: Dict[str, str], device: str = "cuda"):
        # Use ColBERT model via sentence-transformers
        self.model = SentenceTransformer("colbert-ir/colbertv2.0", device=device)
        self.corpus = corpus
        self.device = device

        self.doc_ids = list(corpus.keys())
        self.doc_texts = [corpus[did] for did in self.doc_ids]

        # For benchmarking, we'll use the [CLS] token embedding (simplified)
        # True ColBERT uses late interaction with token-level embeddings
        print("Encoding corpus with ColBERT (simplified)...")
        batch_size = 64
        embeddings = []
        for i in range(0, len(self.doc_texts), batch_size):
            batch = self.doc_texts[i:i+batch_size]
            emb = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False, device=device)
            embeddings.append(emb)
        self.doc_embeddings = np.vstack(embeddings).astype(np.float32)

        faiss.normalize_L2(self.doc_embeddings)
        dim = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.doc_embeddings)
        print(f"ColBERT index built: {self.index.ntotal} vectors")

    def batch_search(self, queries: List[str], k: int) -> List[List[Tuple[str, float]]]:
        query_embeddings = self.model.encode(queries, convert_to_numpy=True, show_progress_bar=False, device=self.device)
        query_embeddings = query_embeddings.astype(np.float32)
        faiss.normalize_L2(query_embeddings)

        scores, indices = self.index.search(query_embeddings, k)

        all_results = []
        for i in range(len(queries)):
            results = []
            for j in range(k):
                idx = indices[i][j]
                if idx >= 0:
                    doc_id = self.doc_ids[idx]
                    score = float(scores[i][j])
                    results.append((doc_id, score))
            all_results.append(results)
        return all_results


# ============================================================
# DATA LOADERS
# ============================================================

def load_msmarco_sample(max_docs: int = 50000, max_queries: int = 1000) -> Dict:
    """Load MS MARCO sample (dev set + corpus subset)."""
    from beir.datasets.data_loader import GenericDataLoader

    # BEIR provides MS MARCO
    data_loader = GenericDataLoader(data_folder="datasets/msmarco")
    corpus, queries, qrels = data_loader.load(split="dev")

    # Subsample
    if len(corpus) > max_docs:
        doc_ids = list(corpus.keys())[:max_docs]
        corpus = {did: corpus[did] for did in doc_ids}
        # Filter qrels
        qrels = {qid: {did: rel for did, rel in docs.items() if did in corpus} for qid, docs in qrels.items()}

    if len(queries) > max_queries:
        query_ids = list(queries.keys())[:max_queries]
        queries = {qid: queries[qid] for qid in query_ids}
        qrels = {qid: qrels[qid] for qid in query_ids if qid in qrels}

    return {"corpus": corpus, "queries": queries, "qrels": qrels}


def load_beir_dataset(name: str, max_queries: int = 500) -> Dict:
    """Load any BEIR dataset."""
    from beir.datasets.data_loader import GenericDataLoader
    import os
    from beir.util import download_and_unzip

    data_path = f"datasets/{name}"
    if not os.path.exists(data_path):
        print(f"Downloading {name}...")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
        download_and_unzip(url, "datasets/")

    data_loader = GenericDataLoader(data_folder=data_path)
    corpus, queries, qrels = data_loader.load(split="test")

    if len(queries) > max_queries:
        query_ids = list(queries.keys())[:max_queries]
        queries = {qid: queries[qid] for qid in query_ids}
        qrels = {qid: qrels[qid] for qid in query_ids if qid in qrels}

    return {"corpus": corpus, "queries": queries, "qrels": qrels}


# ============================================================
# REGISTRY
# ============================================================

RETRIEVER_REGISTRY = {
    "bm25": BM25Retriever,
    "bge-large": DenseRetriever,
    "bge-base": DenseRetriever,
    "splade": SPLADERetriever,
    "colbert": ColBERTRetriever,
}

DATASET_REGISTRY = {
    "msmarco": load_msmarco_sample,
    "nfcorpus": lambda **kw: load_beir_dataset("nfcorpus", **kw),
    "scifact": lambda **kw: load_beir_dataset("scifact", **kw),
    "trec-covid": lambda **kw: load_beir_dataset("trec-covid", **kw),
    "fiqa": lambda **kw: load_beir_dataset("fiqa", **kw),
}


def get_retriever(name: str, corpus: Dict, model_name: str = None, device: str = "cuda") -> BaseRetriever:
    """Factory function to create retriever."""
    cls = RETRIEVER_REGISTRY[name]
    if name in ["bge-large", "bge-base"]:
        model = "BAAI/bge-large-en-v1.5" if "large" in name else "BAAI/bge-base-en-v1.5"
        return cls(model, corpus, device)
    return cls(corpus, device)


if __name__ == "__main__":
    # Quick test
    import json
    test_corpus = {
        "doc1": "Machine learning is a subset of artificial intelligence.",
        "doc2": "Deep learning uses neural networks with many layers.",
        "doc3": "Natural language processing enables computers to understand text."
    }
    queries = ["What is machine learning?", "How do neural networks work?"]

    print("Testing BM25...")
    bm25 = BM25Retriever(test_corpus)
    results = bm25.batch_search(queries, k=2)
    print(results)

    print("\nTesting Dense (BGE-base)...")
    dense = DenseRetriever("BAAI/bge-base-en-v1.5", test_corpus, device="cpu")
    results = dense.batch_search(queries, k=2)
    print(results)