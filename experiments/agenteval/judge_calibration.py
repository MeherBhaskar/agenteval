#!/usr/bin/env python3
"""
Judge Calibration for AgentEval
Compares Nemotron Ultra judges on 100 samples.
Only uses Nemotron Ultra via LiteLLM proxy.
"""

import os
import json
import argparse
import random
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests


LITELLM_BASE = os.environ.get("ANTHROPIC_BASE_URL", "http://localhost:4000")
MODEL = "openai/nvidia/nemotron-3-ultra-550b-a55b"


@dataclass
class JudgeScore:
    sample_id: str
    judge: str
    scores: Dict[str, float]
    reasoning: str


def load_samples(samples_path: Path) -> List[Dict]:
    """Load calibration samples from JSONL."""
    samples = []
    with open(samples_path) as f:
        for line in f:
            data = json.loads(line)
            samples.append(data)
    return samples


def create_prompt(sample: Dict) -> str:
    """Create strict JSON-only prompt for judge."""
    traj_str = json.dumps(sample["trajectory"], indent=2)
    gt_str = json.dumps(sample["ground_truth"], indent=2)

    return f"""You are an automated scoring system. Output ONLY valid JSON. No explanations, no markdown, no text outside the JSON object.

TASK: {sample["task_id"]}
SAMPLE: {sample["sample_id"]}

TRAJECTORY:
{json.dumps(sample["trajectory"], indent=2)}

GROUND TRUTH:
{json.dumps(sample["ground_truth"], indent=2)}

Return ONLY this JSON structure (use 0.0 for irrelevant metrics):
{{
  "scores": {{
    "exact_match": 0.0,
    "f1": 0.0,
    "retrieval_recall@10": 0.0,
    "acceptance_rate": 0.0,
    "margin_retention_pct": 0.0,
    "test_pass_rate": 0.0,
    "api_correctness": 0.0,
    "purchase_success": 0.0,
    "price_optimality": 0.0,
    "query_relevance": 0.0,
    "constraint_satisfaction": 0.0,
    "total_cost_vs_budget": 0.0,
    "preference_alignment": 0.0,
    "itinerary_quality": 0.0
  }},
  "reasoning": "max 20 words"
}}"""


def call_judge(prompt: str) -> Dict:
    """Call Nemotron Ultra via LiteLLM proxy."""
    payload = {
        "model": "openai/nvidia/nemotron-3-ultra-550b-a55b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
        "response_format": {"type": "json_object"}
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(
        "http://localhost:4000/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]

    # Parse JSON strictly - find outermost {}
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(content[start:end])
    raise ValueError(f"Failed to parse JSON from: {content[:200]}")


def judge_sample(sample: Dict) -> Dict:
    """Score one sample with Nemotron Ultra."""
    prompt = f"""You are an automated scoring system. Output ONLY valid JSON. No explanations, no markdown, no text outside the JSON object.

TASK: {sample["task_id"]}
SAMPLE: {sample["sample_id"]}

TRAJECTORY:
{json.dumps(sample["trajectory"], indent=2)}

GROUND TRUTH:
{json.dumps(sample["ground_truth"], indent=2)}

Return ONLY this JSON structure (use 0.0 for irrelevant metrics):
{{
  "scores": {{
    "exact_match": 0.0,
    "f1": 0.0,
    "retrieval_recall@10": 0.0,
    "acceptance_rate": 0.0,
    "margin_retention_pct": 0.0,
    "test_pass_rate": 0.0,
    "api_correctness": 0.0,
    "purchase_success": 0.0,
    "price_optimality": 0.0,
    "query_relevance": 0.0,
    "constraint_satisfaction": 0.0,
    "total_cost_vs_budget": 0.0,
    "preference_alignment": 0.0,
    "itinerary_quality": 0.0
  }},
  "reasoning": "max 20 words"
}}"""

    payload = {
        "model": "openai/nvidia/nemotron-3-ultra-550b-a55b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
        "response_format": {"type": "json_object"}
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(
        "http://localhost:4000/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]

    # Strict JSON parsing
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(content[start:end])
    raise ValueError(f"Failed to parse JSON from: {content[:200]}")


def judge_sample(sample: Dict) -> Dict:
    """Score one sample with Nemotron Ultra."""
    prompt = f"""You are an automated scoring system. Output ONLY valid JSON. No explanations, no markdown, no text outside the JSON object.

TASK: {sample["task_id"]}
SAMPLE: {sample["sample_id"]}

TRAJECTORY:
{json.dumps(sample["trajectory"], indent=2)}

GROUND TRUTH:
{json.dumps(sample["ground_truth"], indent=2)}

Return ONLY this JSON structure (use 0.0 for irrelevant metrics):
{{
  "scores": {{
    "exact_match": 0.0,
    "f1": 0.0,
    "retrieval_recall@10": 0.0,
    "acceptance_rate": 0.0,
    "margin_retention_pct": 0.0,
    "test_pass_rate": 0.0,
    "api_correctness": 0.0,
    "purchase_success": 0.0,
    "price_optimality": 0.0,
    "query_relevance": 0.0,
    "constraint_satisfaction": 0.0,
    "total_cost_vs_budget": 0.0,
    "preference_alignment": 0.0,
    "itinerary_quality": 0.0
  }},
  "reasoning": "max 10 words"
}}"""

    payload = {
        "model": "openai/nvidia/nemotron-3-ultra-550b-a55b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(
        "http://localhost:4000/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]

    # Parse JSON strictly
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(content[start:end])

    raise ValueError(f"Failed to parse JSON from: {content[:200]}")


def main():
    parser = argparse.ArgumentParser(description="Judge Calibration")
    parser.add_argument("--samples", default="calibration_samples.jsonl")
    parser.add_argument("--output", default="results/judge_calibration_results.jsonl")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()

    # Load samples
    with open(args.samples) as f:
        samples = [json.loads(line) for line in f]
    samples = samples[:args.n_samples]

    print(f"Loaded {len(samples)} samples")
    print("Running calibration with Nemotron Ultra...")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, sample in enumerate(samples):
        try:
            print(f"  Sample {sample['sample_id']} ({i+1}/{len(samples)})")
            result = judge_sample(sample)
            print(f"  Scores: {result}")
        except Exception as e:
            print(f"  Error: {e}")
            yield {
                "sample_id": sample["sample_id"],
                "task_id": sample["task_id"],
                "judge": "nemotron-3-ultra",
                "scores": {},
                "reasoning": f"ERROR: {e}"
            }

        # Save incrementally
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    print(f"\nCompleted {len(results)} evaluations")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Judge Calibration")
    parser.add_argument("--samples", default="calibration_samples.jsonl")
    parser.add_argument("--output", default="results/judge_calibration_results.jsonl")
    parser.add_argument("--n-samples", type=int, default=100)
    args = parser.parse_args()

    main()