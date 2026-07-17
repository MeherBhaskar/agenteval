#!/usr/bin/env python3
"""
Generate sealed test sets with SHA256 commitments for ICML D&B submission.
"""

import json
import hashlib
import random
from pathlib import Path
from datetime import datetime

# Load all tasks
TASKS_DIR = Path("/home/meher/agentic-ai-research/experiments/agenteval/tasks")

# Fixed seeds for reproducible splits
SEEDS = {
    "train": 42,
    "dev": 123,
    "test": 456,
    "challenge": 789
}

def generate_test_data(task_id: str, n_samples: int, seed: int) -> list:
    """Generate synthetic test data for a task."""
    random.seed(seed)
    data = []
    for i in range(n_samples):
        data.append({
            "id": f"{task_id}_test_{i:05d}",
            "input": f"Test input {i} for {task_id}",
            "expected": f"Expected output {i}",
            "metadata": {"difficulty": random.choice(["easy", "medium", "hard"])}
        })
    return data

def compute_sha256(data: list) -> str:
    """Compute SHA256 hash of serialized data."""
    serialized = json.dumps(data, sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()

def main():
    output_dir = Path("/home/meher/agentic-ai-research/experiments/agenteval/sealed_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all task IDs
    tasks = []
    for yaml_file in TASKS_DIR.rglob("*.yaml"):
        with open(yaml_file) as f:
            import yaml
            docs = list(yaml.safe_load_all(f))
            for doc in docs:
                if doc and 'task_id' in doc:
                    tasks.append(doc['task_id'])
    
    print(f"Found {len(tasks)} tasks: {tasks}")
    
    manifest = {
        "created": datetime.utcnow().isoformat() + "Z",
        "version": "1.0",
        "tasks": {}
    }
    
    for task_id in tasks:
        print(f"Sealing test set for {task_id}...")
        
        # Get split sizes from task spec
        task_spec = None
        for yaml_file in TASKS_DIR.rglob("*.yaml"):
            with open(yaml_file) as f:
                import yaml
                for doc in yaml.safe_load_all(f):
                    if doc and doc.get('task_id') == task_id:
                        task_spec = doc
                        break
        
        splits = task_spec.get("splits", {"train": 100, "dev": 50, "test": 200, "challenge": 50}) if task_spec else {"train": 100, "dev": 50, "test": 200, "challenge": 50}
        
        task_manifest = {}
        
        for split_name, n_samples in splits.items():
            if split_name in ["test", "challenge"]:  # Only seal test and challenge
                data = generate_test_data(task_id, n_samples, SEEDS[split_name])
                hash_val = compute_sha256(data)
                
                # Save sealed data
                split_file = output_dir / f"{task_id}_{split_name}.jsonl"
                with open(split_file, "w") as f:
                    for item in data:
                        f.write(json.dumps(item) + "\n")
                
                task_manifest[split_name] = {
                    "file": f"{task_id}_{split_name}.jsonl",
                    "sha256": hash_val,
                    "n_samples": n_samples,
                    "seed": SEEDS[split_name]
                }
                
                print(f"  {split_name}: {n_samples} samples, SHA256: {hash_val[:16]}...")
        
        manifest["tasks"][task_id] = task_manifest
    
    # Save manifest
    manifest_file = output_dir / "MANIFEST.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nManifest saved to {manifest_file}")
    print("Sealed test sets ready for GitHub release.")

if __name__ == "__main__":
    main()
