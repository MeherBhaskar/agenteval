#!/usr/bin/env python3
"""
Fact QA Task Server - Multi-hop fact-seeking QA with retrieval and reasoning.
"""
import os
import json
import random
from pathlib import Path
from flask import Flask, request, jsonify

app = Flask(__name__)

TASK_DATA_DIR = Path(os.environ.get("TASK_DATA_DIR", "/app/tasks/fact_qa/data"))

# Load or generate test data
def load_data(split: str):
    """Load or generate data for the given split."""
    data_file = TASK_DATA_DIR / f"{split}.jsonl"
    if data_file.exists():
        with open(data_file) as f:
            return [json.loads(line) for line in f]

    # Generate synthetic data if not exists
    TASK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = []
    for i in range(100 if split != "test" else 200):
        item = {
            "id": f"{split}_{i:04d}",
            "question": f"What is the capital of country X{i}?",
            "answer": f"Capital{i}",
            "supporting_facts": [f"Fact {i}.1", f"Fact {i}.2"],
            "metadata": {"difficulty": random.choice(["easy", "medium", "hard"])}
        }
        data.append(item)

    with open(data_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return data

# Global data cache
DATA_CACHE = {}


class FactQAEnv:
    """Environment for Fact QA task."""

    def __init__(self, split: str = "dev", seed: int = 42):
        self.split = split
        self.seed = seed
        self.current_item = None
        self.step_count = 0
        self.max_steps = 5

        if split not in DATA_CACHE:
            DATA_CACHE[split] = load_data(split)
        self.data = DATA_CACHE[split]
        self.rng = random.Random(seed)

    def reset(self):
        """Start new episode."""
        self.current_item = self.rng.choice(self.data)
        self.step_count = 0
        return {
            "question": self.current_item["question"],
            "step": 0,
            "context": ""
        }

    def step(self, action: dict):
        """Execute one step."""
        self.step_count += 1
        action_type = action.get("action_type", "")
        content = action.get("content", "")

        reward = 0.0
        done = False
        info = {}

        if action_type == "search":
            # Simulate retrieval
            reward = 0.2
            context = f"Retrieved: {self.current_item['supporting_facts'][0]}"
        elif action_type == "reason":
            # Simulate reasoning
            reward = 0.3
            context = f"Reasoning step: {content}"
        elif action_type == "answer":
            # Check answer
            if content.lower().strip() == self.current_item["answer"].lower().strip():
                reward = 1.0
            else:
                reward = 0.0
            done = True
            info = {"correct": reward > 0, "expected": self.current_item["answer"]}
            context = ""
        else:
            context = "Unknown action type"

        return {
            "observation": {
                "question": self.current_item["question"],
                "step": self.step_count,
                "context": context
            },
            "reward": reward,
            "done": done or self.step_count >= self.max_steps,
            "info": info
        }


# Global environment instance
_env = None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/task", methods=["GET"])
def get_task():
    with open("/app/tasks/fact_qa/task.json") as f:
        return jsonify(json.load(f))


@app.route("/reset", methods=["POST"])
def reset():
    global _env
    data = request.get_json() or {}
    split = data.get("split", "dev")
    seed = data.get("seed", 42)
    _env = FactQAEnv(split=split, seed=seed)
    obs = _env.reset()
    return jsonify(obs)


@app.route("/step", methods=["POST"])
def step():
    global _env
    if _env is None:
        return jsonify({"error": "Environment not initialized. Call /reset first."}), 400

    action = request.get_json() or {}
    result = _env.step(action)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)