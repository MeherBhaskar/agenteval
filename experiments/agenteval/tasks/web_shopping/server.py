#!/usr/bin/env python3
"""
Web Shopping Task Server - Find and purchase products on simulated e-commerce site.
"""
import os
import json
import random
from pathlib import Path
from flask import Flask, request, jsonify

app = Flask(__name__)

TASK_DATA_DIR = Path(os.environ.get("TASK_DATA_DIR", "/app/tasks/web_shopping/data"))


def load_data(split: str):
    data_file = TASK_DATA_DIR / f"{split}.jsonl"
    if data_file.exists():
        with open(data_file) as f:
            return [json.loads(line) for f in f]

    TASK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    categories = ["electronics", "clothing", "home", "books", "sports", "beauty"]
    products = []
    for i in range(5000):
        products.append({
            "id": f"PROD_{i:05d}",
            "name": f"Product {i}",
            "category": random.choice(categories),
            "price": round(random.uniform(5, 500), 2),
            "rating": round(random.uniform(3.0, 5.0), 1),
            "in_stock": random.choice([True, True, True, False])  # 75% in stock
        })

    data = []
    for i in range(100 if split != "test" else 200):
        query_products = random.sample(products, 20)
        target = random.choice([p for p in query_products if p["in_stock"]])
        item = {
            "id": f"{split}_{i:04d}",
            "query": f"Find {target['category']} under ${target['price'] + 20:.0f}",
            "target_product": target["id"],
            "max_price": round(target["price"] + random.uniform(5, 20), 2),
            "category": target["category"],
            "products": [p for p in query_products if p["category"] == target["category"]],
            "ground_truth": {
                "target_id": target["id"],
                "target_price": target["price"],
                "optimal_price": target["price"]
            }
        }
        data.append(item)

    with open(data_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return data


DATA_CACHE = {}


class WebShoppingEnv:
    def __init__(self, split: str = "dev", seed: int = 42):
        self.split = split
        self.seed = seed
        self.current_item = None
        self.cart = []
        self.current_page = "search"

        if split not in DATA_CACHE:
            DATA_CACHE[split] = load_data(split)
        self.data = DATA_CACHE[split]
        self.rng = random.Random(seed)

    def reset(self):
        self.current_item = self.rng.choice(self.data)
        self.cart = []
        self.current_page = "search"
        return {
            "query": self.current_item["query"],
            "page": "search_results",
            "products": self.current_item["products"][:10],
            "cart": [],
            "url": "https://shop.example.com/search",
            "ground_truth": self.current_item["ground_truth"]
        }

    def step(self, action: dict):
        action_type = action.get("action_type", "")
        selector = action.get("selector", "")
        value = action.get("value", "")

        reward = 0.0
        done = False
        info = {}

        if action_type == "search":
            # Filter products
            query = value.lower()
            filtered = [p for p in self.current_item["products"]
                       if query in p["name"].lower() or query in p["category"].lower()]
            self.current_page = "search_results"
            reward = 0.1

        elif action_type == "click":
            # Click product
            if selector.startswith("PROD_"):
                product = next((p for p in self.current_item["products"] if p["id"] == selector), None)
                if product:
                    self.current_page = "product"
                    reward = 0.2
                else:
                    reward = -0.1

        elif action_type == "buy":
            # Attempt purchase
            if self.current_page == "product":
                # Check if it's the target product
                if selector == self.current_item["ground_truth"]["target_id"]:
                    reward = 1.0
                    info["purchased"] = True
                    info["price_optimal"] = True
                else:
                    reward = 0.3
                    info["purchased"] = True
                    info["price_optimal"] = False
                done = True
            else:
                reward = -0.1

        elif action_type == "add_to_cart":
            if selector.startswith("PROD_"):
                self.cart.append(selector)
                reward = 0.1
                done = False

        else:
            reward = -0.1

        return {
            "observation": {
                "query": self.current_item["query"],
                "page": self.current_page,
                "products": self.current_item["products"][:10] if self.current_page == "search_results" else [],
                "cart": self.cart,
                "url": f"https://shop.example.com/{self.current_page}",
                "ground_truth": self.current_item["ground_truth"]
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
    with open("/app/tasks/web_shopping/task.json") as f:
        return jsonify(json.load(f))


@app.route("/reset", methods=["POST"])
def reset():
    global _env
    data = request.get_json() or {}
    split = data.get("split", "dev")
    seed = data.get("seed", 42)
    _env = WebShoppingEnv(split=split, seed=seed)
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