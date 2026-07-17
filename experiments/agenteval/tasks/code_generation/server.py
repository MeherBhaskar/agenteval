#!/usr/bin/env python3
"""
Code Generation Task Server - Generate working code using specified API/library.
"""
import os
import json
import subprocess
import tempfile
import random
from pathlib import Path
from flask import Flask, request, jsonify

app = Flask(__name__)

TASK_DATA_DIR = Path(os.environ.get("TASK_DATA_DIR", "/app/tasks/code_generation/data"))


def load_data(split: str):
    data_file = Path("/app/tasks/code_generation/data") / f"{split}.jsonl"
    if data_file.exists():
        with open(data_file) as f:
            return [json.loads(line) for line in f]

    # Generate synthetic tasks
    tasks = [
        {"description": "Fetch data from REST API", "api": "requests", "target": "GET /users returns JSON array"},
        {"description": "Parse JSON and extract fields", "api": "json", "target": "extract name, email from user objects"},
        {"description": "Write data to CSV file", "api": "csv", "target": "write list of dicts to CSV"},
        {"description": "Read file and count lines", "api": "builtins", "target": "count lines in text file"},
        {"description": "Make HTTP POST with JSON body", "api": "requests", "target": "POST /api/data with JSON"},
        {"description": "Parse HTML and extract links", "api": "BeautifulSoup", "target": "find all <a> tags"},
        {"description": "Validate email format", "api": "re", "target": "email regex validation"},
        {"description": "Calculate file hash", "api": "hashlib", "target": "SHA256 of file contents"},
        {"description": "Compress data with gzip", "api": "gzip", "target": "compress string data"},
        {"description": "Parse date string", "api": "datetime", "target": "parse ISO format date"},
    ]

    TASK_DATA_DIR = Path("/app/tasks/code_generation/data")
    TASK_DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = []
    for i, task in enumerate(tasks):
        item = {
            "id": f"code_{i:04d}",
            "task_description": task["description"],
            "api_docs": f"Standard {task['api']} library documentation",
            "example_usage": f"import {task['api']}\n# {task['target']}",
            "constraints": ["handle errors", "no external dependencies beyond stdlib"],
            "ground_truth": {
                "expected_output": f"Working code that {task['target'].lower()}",
                "required_imports": [task["api"]],
                "test_cases": [{"input": "sample", "expected": "success"}]
            }
        }
        data.append(item)

    Path("/app/tasks/code_generation/data").mkdir(parents=True, exist_ok=True)
    with open("/app/tasks/code_generation/data/train.jsonl", "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    return data[:100]


DATA_CACHE = {}


class CodeGenEnv:
    def __init__(self, split: str = "dev", seed: int = 42):
        self.split = split
        self.seed = seed
        self.current_task = None

        if split not in DATA_CACHE:
            DATA_CACHE[split] = load_data(split)
        self.data = DATA_CACHE[split]
        self.rng = random.Random(seed)

    def reset(self):
        self.current_task = self.rng.choice(self.data)
        return {
            "task_description": self.current_task["task_description"],
            "api_docs": self.current_task["api_docs"],
            "example_usage": self.current_task["example_usage"],
            "constraints": self.current_task["constraints"]
        }

    def step(self, action: dict):
        code = action.get("code", "")
        explanation = action.get("explanation", "")

        # Write code to temp file and run tests
        reward = 0.0
        done = True
        info = {}

        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            # Run basic syntax check
            result = subprocess.run(
                ["python3", "-m", "py_compile", temp_path],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                reward += 0.3
                info["syntax_ok"] = True
                info["syntax_ok"] = True
            else:
                info["syntax_error"] = result.stderr

            # Check if required imports present
            gt = self.current_task.get("ground_truth", {})
            required = gt.get("required_imports", [])
            imports_ok = all(imp in code for imp in required)
            if imports_ok:
                reward += 0.2
                info["imports_ok"] = True

            # Run any test cases if they exist
            # (simplified - just check code runs without error)
            test_code = code + "\n\n# Test execution\nprint('Code executed successfully')"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_code)
                test_path = f.name

            result = subprocess.run(
                ["python3", test_path],
                capture_output=True, text=True, timeout=10
            )
            os.unlink(test_path)

            if result.returncode == 0:
                reward += 0.5
                info["execution_ok"] = True
            else:
                info["execution_error"] = result.stderr[:200]

        except subprocess.TimeoutExpired:
            info["error"] = "Timeout"
        except Exception as e:
            info["error"] = str(e)
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass

        return {
            "observation": {"result": "Code evaluated", "reward": reward},
            "reward": max(0, min(1, reward)),
            "done": True,
            "info": info
        }


_env = None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/task", methods=["GET"])
def get_task():
    with open("/app/tasks/code_generation/task.json") as f:
        return jsonify(json.load(f))


@app.route("/reset", methods=["POST"])
def reset():
    global _env
    data = request.get_json() or {}
    split = data.get("split", "dev")
    seed = data.get("seed", 42)
    _env = CodeGenEnv(split=split, seed=seed)
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