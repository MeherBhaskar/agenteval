#!/usr/bin/env python3
"""
Critique Model Data Generation for Paper 1 (Critique-Loop RAG)
Generates synthetic critique data using LiteLLM proxy (Nemotron Ultra).
"""

import os
import json
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests


# ============================================================
# CONFIGURATION
# ============================================================

LITELLM_BASE = os.environ.get("ANTHROPIC_BASE_URL", "http://localhost:4000")
LITELLM_MODEL = os.environ.get("ANTHROPIC_CUSTOM_MODEL_OPTION", "nemotron-ultra")

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
"Hallucination spans" must be exact substrings from the answer.
"""

# Test queries with pre-retrieved docs
TEST_QUERIES = [
    {
        "id": "test_001",
        "query": "What are the health benefits of green tea?",
        "docs": [
            {"id": "doc_1", "text": "Green tea contains catechins, particularly EGCG, which are powerful antioxidants that may reduce inflammation and cell damage."},
            {"id": "doc_2", "text": "Studies show green tea consumption is associated with lower risk of cardiovascular disease and improved cholesterol levels."},
            {"id": "doc_3", "text": "The caffeine and L-theanine in green tea can improve brain function, including mood, vigilance, and reaction time."},
            {"id": "doc_4", "text": "Green tea extract has been shown to boost metabolic rate and increase fat burning in short-term studies."},
            {"id": "doc_5", "text": "Coffee also contains antioxidants but has higher caffeine content which may cause jitters in sensitive individuals."}
        ]
    },
    {
        "id": "test_002",
        "query": "How does a blockchain work?",
        "docs": [
            {"id": "doc_1", "text": "A blockchain is a distributed ledger that records transactions across a network of computers."},
            {"id": "doc_2", "text": "Each block contains a cryptographic hash of the previous block, a timestamp, and transaction data."},
            {"id": "doc_3", "text": "Proof of Work requires miners to solve complex mathematical puzzles to validate blocks."},
            {"id": "doc_4", "text": "Smart contracts are self-executing contracts with terms directly written into code on the blockchain."},
            {"id": "doc_5", "text": "Bitcoin was the first cryptocurrency to use blockchain technology, created in 2009 by Satoshi Nakamoto."}
        ]
    },
    {
        "id": "test_003",
        "query": "What causes climate change?",
        "docs": [
            {"id": "doc_1", "text": "The primary cause of climate change is the increase in greenhouse gases from human activities, especially burning fossil fuels."},
            {"id": "doc_2", "text": "Carbon dioxide levels have increased by over 50% since pre-industrial times, trapping more heat in the atmosphere."},
            {"id": "doc_3", "text": "Deforestation reduces the planet's capacity to absorb CO2, contributing to rising atmospheric concentrations."},
            {"id": "doc_4", "text": "Methane from agriculture and landfills is a potent greenhouse gas with 25x the warming potential of CO2."},
            {"id": "doc_5", "text": "Natural climate cycles like Milankovitch cycles operate over tens of thousands of years and don't explain recent warming."}
        ]
    }
]


@dataclass
class CritiqueExample:
    query: str
    query_id: str
    retrieved_docs: List[Dict[str, Any]]
    generated_answer: str
    critique: Dict[str, Any]
    retriever_used: str
    generator_used: str


class NemotronClient:
    """Client for Nemotron Ultra via LiteLLM proxy."""

    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or LITELLM_BASE
        self.model = model or LITELLM_MODEL
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('ANTHROPIC_API_KEY', 'not-used')}"
        }

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        """Call the LiteLLM proxy with Anthropic-compatible format."""
        # LiteLLM expects OpenAI format but we use Anthropic-style env vars
        # The proxy handles the conversion
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95
        }

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def generate_answer(client: NemotronClient, query: str, docs: List[Dict]) -> str:
    """Generate an answer using the retrieved docs."""
    doc_text = "\n\n".join([f"[{d['id']}] {d['text']}" for d in docs])

    prompt = f"""Answer the question using ONLY the provided documents. If the answer isn't in the docs, say "I don't know based on the provided documents."

QUESTION: {query}

DOCUMENTS:
{doc_text}

ANSWER:"""

    return client.generate(prompt, max_tokens=512, temperature=0.3).strip()


def generate_critique(client: NemotronClient, query: str, docs: List[Dict], answer: str) -> Dict:
    """Generate structured critique."""
    doc_text = "\n\n".join([f"[{d['id']}] {d['text']}" for d in docs])
    prompt = CRITIQUE_PROMPT.format(query=query, documents=doc_text, answer=answer)

    response = client.generate(prompt, max_tokens=1024, temperature=0.1)

    # Parse JSON from response - handle case where model outputs reasoning before JSON
    try:
        # Find the first { and last }
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            return json.loads(json_str)
    except Exception as e:
        print(f"  ⚠️  Failed to parse critique: {e}")
        print(f"  Response: {response[:200]}...")
    return {
        "missing_info": [],
        "irrelevant_docs": [],
        "hallucination_spans": [],
        "quality_score": 3,
        "reasoning": "Parse error"
    }


def process_query(client: NemotronClient, query_data: Dict) -> CritiqueExample:
    """Process a single query: generate answer, then critique."""
    query = query_data["query"]
    qid = query_data["id"]
    docs = query_data["docs"]

    # Generate answer
    answer = generate_answer(client, query, docs)

    # Generate critique
    critique = generate_critique(client, query, docs, answer)

    return CritiqueExample(
        query=query,
        query_id=qid,
        retrieved_docs=docs,
        generated_answer=answer,
        critique=critique,
        retriever_used="bm25_test",
        generator_used="nemotron-ultra"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate critique training data")
    parser.add_argument("--output", default="experiments/critique_data/critique_train.jsonl")
    parser.add_argument("--n-workers", type=int, default=3)
    parser.add_argument("--test-only", action="store_true", help="Run on 3 test queries only")
    args = parser.parse_args()

    # Check LiteLLM proxy
    try:
        response = requests.get(f"{LITELLM_BASE}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ LiteLLM proxy not healthy: {response.status_code}")
            print("Start it with: source ~/litellm_env/bin/activate && litellm --config ~/litellm_config.yaml")
            return
    except Exception as e:
        print(f"❌ Cannot connect to LiteLLM proxy at {LITELLM_BASE}: {e}")
        print("Make sure the proxy is running (see claude_nemotron.sh)")
        return

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    client = NemotronClient()

    queries = TEST_QUERIES

    print(f"🚀 Generating critiques for {len(queries)} queries using {args.n_workers} workers...")

    examples = []
    with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(process_query, client, q): q for q in queries}
        for future in tqdm(as_completed(futures), total=len(futures)):
            q = futures[future]
            try:
                example = future.result()
                examples.append(example)
                print(f"\n  ✅ [{q['id']}] Score: {example.critique['quality_score']}/5")
                print(f"  Answer: {example.generated_answer[:100]}...")
                print(f"  Missing: {example.critique['missing_info']}")
            except Exception as e:
                print(f"  ❌ ERROR on {q['id']}: {e}")

    # Save
    with open(args.output, "w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")

    print(f"\n✅ Saved {len(examples)} examples to {args.output}")

    # Quick stats
    scores = [e.critique["quality_score"] for e in examples]
    if scores:
        from collections import Counter
        print(f"Quality distribution: {dict(Counter(scores))}")


if __name__ == "__main__":
    main()