#!/usr/bin/env python3
"""
Debugging Task Server
"""
import os
import json
import random
from pathlib import Path
from flask import Flask, request, jsonify

app = Flask(__name__)

TASK_DATA_DIR = Path(os.environ.get("TASK_DATA_DIR", "/app/tasks/debugging/data"))


def load_data(split: str):
    data_file = TASK_DATA_DIR / f"{split}.jsonl"
    if data_file.exists():
        with open(data_file) as f:
            return [json.loads(line) for line in f]
    return []


DATA_CACHE = {}


class DebuggingEnv:
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
        self.current_item = self.rng.choice(self.data) if self.data else {}
        self.step_count = 0
        return {
            "input": self.current_item.get("input", ""),
            "step": 0
        }

    def step(self, action: dict):
        self.step_count += 1
        action_type = action.get("action", action.get("output", ""))
        
        reward = 0.0
        done = False
        info = {}

        if action_type:
            expected = self.current_item.get("output", "")
            given = str(action.get("output", action.get("action", "")))
            
            if expected.lower() in given.lower() or given.lower() in expected.lower():
                reward = 1.0
                info["correct"] = True
            else:
                reward = 0.1
                info["correct"] = False
            done = True
            info["expected"] = expected
            info["given"] = given

        return {
            "observation": {
                "input": self.current_item.get("input", ""),
                "step": self.step_count
            },
            "reward": reward,
            "done": done or self.step_count >= self.max_steps,
            "info": info
        }


_env = None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/task", methods=["GET"])
def get_task():
    with open("/app/tasks/debugging/task.json") as f:
        return jsonify(json.load(f))


@app.route("/reset", methods=["POST"])
def reset():
    global _env
    data = request.get_json() or {}
    split = data.get("split", "dev")
    seed = data.get("seed", 42)
    _env = DebuggingEnv(split=split, seed=seed)
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
