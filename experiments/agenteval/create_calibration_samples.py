#!/usr/bin/env python3
"""
Create 100 calibration samples for judge calibration.
"""

import json
import random
from pathlib import Path

# Task categories and their specific metrics
TASK_TYPES = {
    "fact_qa": ["exact_match", "f1", "retrieval_recall@10"],
    "retail": ["acceptance_rate", "margin_retention_pct"],
    "code_gen": ["test_pass_rate", "api_correctness"],
    "web_shopping": ["purchase_success", "price_optimality", "query_relevance"],
    "travel": ["constraint_satisfaction", "total_cost_vs_budget", "preference_alignment"],
    "data_analysis": ["output_match", "code_quality_score"],
    "sentiment": ["accuracy", "f1"],
    "summarization": ["rouge_l", "factual_consistency"],
    "api_integration": ["success_rate", "latency_ms"],
    "qa": ["exact_match", "f1"],
    "code_review": ["bug_detection", "suggestion_quality"],
    "translation": ["bleu", "comet"],
    "sql_generation": ["execution_accuracy", "syntax_correctness"],
    "entity_extraction": ["precision", "recall", "f1"],
    "debugging": ["bug_fixed", "test_pass"],
    "text_generation": ["coherence", "relevance", "fluency"]
}

def create_sample(sample_id: int, task_type: str, quality: str):
    """Create a calibration sample with specified quality level."""
    task_id = f"{task_type}_01"
    metrics = TASK_TYPES[task_type]

    if quality == "excellent":
        score_base = 0.9 + random.uniform(0, 0.1)
        trajectory = [{"step": i, "observation": f"step {i}", "action": f"correct_action_{i}"} for i in range(3)]
    elif quality == "good":
        score_base = 0.7 + random.uniform(0, 0.15)
        trajectory = [{"step": i, "observation": f"step {i}", "action": f"mostly_correct_{i}"} for i in range(4)]
    elif quality == "fair":
        score_base = 0.5 + random.uniform(0, 0.15)
        trajectory = [{"step": i, "observation": f"step {i}", "action": f"partial_{i}"} for i in range(3)]
    else:  # poor
        score_base = 0.1 + random.uniform(0, 0.2)
        trajectory = [{"step": i, "observation": f"step {i}", "action": f"wrong_{i}"} for i in range(2)]

    ground_truth = {m: round(min(1.0, max(0.0, score_base + random.uniform(-0.05, 0.05))), 2) for m in metrics}

    return {
        "sample_id": f"cal_{sample_id:03d}",
        "task_id": task_id,
        "trajectory": trajectory,
        "ground_truth": ground_truth
    }

def main():
    random.seed(42)

    # Distribute samples across task types and quality levels
    qualities = ["excellent"] * 25 + ["good"] * 30 + ["fair"] * 25 + ["poor"] * 20
    random.shuffle(qualities)

    samples = []
    for i, quality in enumerate(qualities):
        task_type = random.choice(list(TASK_TYPES.keys()))
        samples.append(create_sample(i, task_type, quality))

    output_path = Path("/home/meher/agentic-ai-research/experiments/agenteval/calibration_samples_100.jsonl")
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Created {len(samples)} calibration samples at {output_path}")

    # Print distribution
    from collections import Counter
    task_counts = Counter(s["task_id"] for s in samples)
    print(f"Task distribution: {dict(task_counts)}")
    print(f"Quality distribution: {Counter(qualities)}")

if __name__ == "__main__":
    main()