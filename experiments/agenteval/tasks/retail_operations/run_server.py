#!/usr/bin/env python3
"""
Retail Operations task server for AgentEval.
"""
import json
import os
from typing import Dict, List, Any
from flask import Flask, request, jsonify

app = Flask(__name__)

TASK_DATA_PATH = os.environ.get("TASK_DATA_PATH", "/app/tasks/retail_operations")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/task", methods=["GET"])
def get_task():
    """Return task specification"""
    with open(os.path.join(TASK_DATA_PATH, "task.json")) as f:
        return jsonify(json.load(f))

@app.route("/step", methods=["POST"])
def step():
    """Execute one step of the task"""
    data = request.json
    return jsonify({
        "observation": "Task running...",
        "reward": 0.0,
        "done": False,
        "info": {}
    })

@app.route("/reset", methods=["POST"])
def reset():
    """Reset the task"""
    return jsonify({"observation": "Task reset", "reward": 0.0, "done": False})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
