#!/usr/bin/env python3
"""
Judge Calibration for AgentEval
Compares GPT-4o, Nemotron Ultra, and Human judges on 100 samples.
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
from scipy.stats import spearmanr


LITELLM_BASE = os.environ.get("ANTHROPIC_BASE_URL", "http://localhost:4000")


@dataclass
class JudgeScore:
    sample_id: str
    judge: str
    scores: Dict[str, float]
    reasoning: str


@dataclass
class CalibrationSample:
    sample_id: str
    task_id: str
    trajectory: List[Dict]
    ground_truth: Dict[str, Any]


def load_ground_truth(samples_path: Path) -> List[CalibrationSample]:
    """Load calibration samples from JSONL."""
    samples = []
    with open(samples_path) as f:
        for line in f:
            data = json.loads(line)
            samples.append(CalibrationSample(
                sample_id=data["sample_id"],
                task_id=data["task_id"],
                trajectory=data["trajectory"],
                ground_truth=data["ground_truth"]
            ))
    return samples


def create_judge_prompt(sample: CalibrationSample) -> str:
    """Create strict JSON-only prompt for judge."""
    traj_str = json.dumps(sample.trajectory, indent=2)
    gt_str = json.dumps(sample.ground_truth, indent=2)
    
    return f"""You are an automated scoring system. Output ONLY valid JSON. No explanations, no markdown, no text outside the JSON object.

TASK: {sample.task_id}
SAMPLE: {sample.sample_id}

TRAJECTORY:
{traj_str}

GROUND TRUTH:
{gt_str}

Return ONLY this JSON structure (use 0.0 for irrelevant metrics):
{{
  "scores": {{
    "exact_match": 0.0,
    "f1": 0.0,
    "retrieval_recall@10": 0.0,
    "acceptance_rate": 0.0,
    "margin_retention_pct": 0.0,
    "test_pass_rate": 0.0,
    "purchase_success": 0.0,
    "constraint_satisfaction": 0.0
  }},
  "reasoning": "max 10 words"
}}"""


def call_judge(prompt: str, model: str) -> Dict:
    """Call LLM judge via LiteLLM."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0
    }
    
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(
        f"{LITELLM_BASE}/v1/chat/completions",
        headers=headers,
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


def judge_sample(sample: CalibrationSample, judge_model: str) -> JudgeScore:
    """Score one sample with one judge."""
    prompt = create_judge_prompt(sample)
    result = call_judge(prompt, model=judge_model)
    
    return JudgeScore(
        sample_id=sample.sample_id,
        judge=judge_model,
        scores=result.get("scores", {}),
        reasoning=result.get("reasoning", "")
    )


def run_calibration(samples: List[CalibrationSample], judges: List[str], n_workers: int = 2) -> List[JudgeScore]:
    """Run calibration across all samples and judges."""
    all_results = []
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for sample in samples:
            for judge in judges:
                future = executor.submit(judge_sample, sample, judge)
                futures[future] = (sample.sample_id, judge)
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Judging"):
            sample_id, judge = futures[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"Error on {sample_id} with {judge}: {e}")
                all_results.append(JudgeScore(
                    sample_id=sample_id,
                    judge=judge,
                    scores={},
                    reasoning=f"ERROR: {e}"
                ))
    
    return all_results


def compute_agreement(results: List[JudgeScore]) -> Dict:
    """Compute inter-judge agreement statistics."""
    by_sample = {}
    for r in results:
        if r.sample_id not in by_sample:
            by_sample[r.sample_id] = {}
        by_sample[r.sample_id][r.judge] = r.scores
    
    all_metrics = set()
    for r in results:
        all_metrics.update(r.scores.keys())
    
    agreement = {}
    judges = list(set(r.judge for r in results))
    
    for metric in all_metrics:
        metric_data = {}
        for sample_id, judge_scores in by_sample.items():
            if all(j in judge_scores and metric in judge_scores[j] for j in judges):
                metric_data[sample_id] = {j: judge_scores[j][metric] for j in judges}
        
        if len(metric_data) < 2:
            continue
            
        pairs = []
        for i, j1 in enumerate(judges):
            for j2 in judges[i+1:]:
                scores1 = [metric_data[s][j1] for s in metric_data]
                scores2 = [metric_data[s][j2] for s in metric_data]
                rho, p = spearmanr(scores1, scores2)
                pairs.append({
                    "judge_1": j1,
                    "judge_2": j2,
                    "spearman_rho": rho,
                    "p_value": p,
                    "n_samples": len(scores1)
                })
        
        agreement[metric] = pairs
    
    return agreement


def main():
    parser = argparse.ArgumentParser(description="Judge Calibration")
    parser.add_argument("--samples", default="calibration_samples.jsonl")
    parser.add_argument("--output", default="results/judge_calibration_results.jsonl")
    parser.add_argument("--judges", nargs="+", default=["gpt-4o", "openai/nvidia/nemotron-3-ultra-550b-a55b"])
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-workers", type=int, default=2)
    args = parser.parse_args()
    
    samples = load_ground_truth(Path(args.samples))
    samples = samples[:args.n_samples]
    
    print(f"Loaded {len(samples)} samples")
    print(f"Judges: {args.judges}")
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    results = run_calibration(samples, args.judges, args.n_workers)
    
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    
    agreement = compute_agreement(results)
    
    print("\n=== INTER-JUDGE AGREEMENT ===")
    for metric, pairs in agreement.items():
        print(f"\n{metric}:")
        for p in pairs:
            print(f"  {p['judge_1']} vs {p['judge_2']}: ρ={p['spearman_rho']:.3f} (p={p['p_value']:.3f}, n={p['n_samples']})")
    
    agreement_path = Path(args.output).with_suffix(".agreement.json")
    with open(agreement_path, "w") as f:
        json.dump(agreement, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"Agreement saved to {agreement_path}")


if __name__ == "__main__":
    main()
