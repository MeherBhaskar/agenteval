#!/usr/bin/env python3
"""
Retail Operations Task Server - Product substitution for out-of-stock items.
"""
import os
import json
import random
from pathlib import Path
from flask import Flask, request, jsonify

app = Flask(__name__)

TASK_DATA_DIR = Path(os.environ.get("TASK_DATA_DIR", "/app/tasks/retail_operations/data"))


def load_data(split: str):
    data_file = TASK_DATA_DIR / f"{split}.jsonl"
    if data_file.exists():
        with open(data_file) as f:
            return [json.loads(line) for line in f]

    TASK_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Product catalog
    products = {
        "MILK_1L": {"category": "dairy", "price": 3.50, "margin": 0.25},
        "MILK_2L": {"category": "dairy", "price": 5.50, "margin": 0.22},
        "MILK_500ML": {"category": "dairy", "price": 2.00, "margin": 0.30},
        "BREAD_WHITE": {"category": "bakery", "price": 2.50, "margin": 0.35},
        "BREAD_WHEAT": {"category": "bakery", "price": 3.00, "margin": 0.30},
        "EGGS_12": {"category": "dairy", "price": 4.00, "margin": 0.20},
        "YOGURT_GREEK": {"category": "dairy", "price": 1.50, "margin": 0.40},
        "CHEESE_CHEDDAR": {"category": "dairy", "price": 5.00, "margin": 0.28},
    }

    # Similarity graph
    similarity = {
        "MILK_1L": [("MILK_2L", 0.9), ("MILK_500ML", 0.7), ("YOGURT_GREEK", 0.4)],
        "MILK_2L": [("MILK_1L", 0.9), ("MILK_500ML", 0.5)],
        "BREAD_WHITE": [("BREAD_WHEAT", 0.8)],
        "EGGS_12": [("YOGURT_GREEK", 0.3)],
    }

    data = []
    for i in range(1000 if split != "test" else 500):
        oos_sku = random.choice(list(products.keys()))
        customer_prefs = {
            "organic": random.random() < 0.3,
            "price_sensitive": random.random() < 0.5,
            "brand_loyal": random.random() < 0.4
        }

        # Generate candidates
        candidates = similarity.get(oos_sku, [])
        if not candidates:
            candidates = [(sku, 0.5) for sku in products if sku != oos_sku][:3]

        # Find best substitute (highest similarity * margin)
        best = max(candidates, key=lambda x: x[1] * products.get(x[0], {}).get("margin", 0))
        best_sku = best[0]

        item = {
            "id": f"{split}_{i:04d}",
            "oos_sku": oos_sku,
            "customer_id": f"C{random.randint(1000, 9999)}",
            "basket": [oos_sku] + random.sample([p for p in products if p != oos_sku], 2),
            "store_inventory": {sku: random.randint(0, 50) for sku in products},
            "similarity_graph": {oos_sku: candidates},
            "customer_preferences": customer_prefs,
            "ground_truth": {
                "best_substitute": best_sku,
                "acceptance_prob": min(0.9, best[1] + random.uniform(0, 0.1)),
                "margin_retention": products[best_sku]["margin"] / products[oos_sku]["margin"] if oos_sku in products else 1.0
            }
        }
        data.append(item)

    with open(data_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return data


DATA_CACHE = {}


class RetailEnv:
    def __init__(self, split: str = "dev", seed: int = 42):
        self.split = split
        self.seed = seed
        self.current_item = None

        if split not in DATA_CACHE:
            DATA_CACHE[split] = load_data(split)
        self.data = DATA_CACHE[split]
        self.rng = random.Random(seed)

    def reset(self):
        self.current_item = self.rng.choice(self.data)
        return {
            "oos_sku": self.current_item["oos_sku"],
            "customer_id": self.current_item["customer_id"],
            "basket": self.current_item["basket"],
            "store_inventory": self.current_item["store_inventory"],
            "similarity_graph": self.current_item["similarity_graph"],
            "customer_preferences": self.current_item["customer_preferences"]
        }

    def step(self, action: dict):
        action_type = action.get("action", "")
        substitute_sku = action.get("substitute_sku", "")
        confidence = action.get("confidence", 0.5)

        reward = 0.0
        done = False
        info = {}

        gt = self.current_item["ground_truth"]

        if action_type == "suggest":
            if substitute_sku == gt["best_substitute"]:
                # Correct substitution
                reward = gt["acceptance_prob"] * gt["margin_retention"]
                info["accepted"] = True
                info["correct"] = True
            elif substitute_sku in self.current_item["similarity_graph"].get(self.current_item["oos_sku"], []):
                # Related but not best
                sim = dict(self.current_item["similarity_graph"][self.current_item["oos_sku"]]).get(substitute_sku, 0.3)
                reward = gt["acceptance_prob"] * 0.5 * sim
                info["accepted"] = random.random() < sim
                info["correct"] = False
            else:
                # Unrelated substitution
                reward = 0.1
                info["accepted"] = False
                info["correct"] = False

            info["suggested"] = substitute_sku
            info["expected"] = gt["best_substitute"]
            info["acceptance_prob"] = gt["acceptance_prob"]
            info["margin_retention"] = gt["margin_retention"]
            done = True

        elif action_type == "auto_sub":
            # Auto-substitute with highest confidence
            reward = gt["acceptance_prob"] * gt["margin_retention"] * confidence
            info["accepted"] = True
            done = True

        elif action_type == "contact":
            # Contact customer - low reward but safe
            reward = 0.3
            info["contacted"] = True
            done = True

        else:
            reward = 0.0
            done = True

        return {
            "observation": {
                "oos_sku": self.current_item["oos_sku"],
                "customer_id": self.current_item["customer_id"],
                "basket": self.current_item["basket"],
                "store_inventory": self.current_item["store_inventory"],
                "similarity_graph": self.current_item["similarity_graph"],
                "customer_preferences": self.current_item["customer_preferences"]
            },
            "reward": max(0, min(1, reward)),
            "done": done,
            "info": info
        }


_env = None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/task", methods=["GET"])
def get_task():
    with open("/app/tasks/retail_operations/task.json") as f:
        return jsonify(json.load(f))


@app.route("/reset", methods=["POST"])
def reset():
    global _env
    data = request.get_json() or {}
    split = data.get("split", "dev")
    seed = data.get("seed", 42)
    _env = RetailEnv(split=split, seed=seed)
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