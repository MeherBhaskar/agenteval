#!/usr/bin/env python3
"""
Travel Planning Task Server - Plan trips with multiple constraints.
"""
import os
import json
import random
from pathlib import Path
from flask import Flask, request, jsonify

app = Flask(__name__)

TASK_DATA_DIR = Path(os.environ.get("TASK_DATA_DIR", "/app/tasks/travel_planning/data"))


def load_data(split: str):
    data_file = TASK_DATA_DIR / f"{split}.jsonl"
    if data_file.exists():
        with open(data_file) as f:
            return [json.loads(line) for line in f]

    TASK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cities = ["Paris", "Tokyo", "New York", "London", "Rome", "Barcelona", "Amsterdam", "Berlin", "Vienna", "Prague"]
    airlines = ["Air France", "Lufthansa", "British Airways", "Delta", "United", "Emirates", "Qatar Airways", "Singapore Airlines"]
    hotels = ["Hilton", "Marriott", "Hyatt", "InterContinental", "Four Seasons", "Ritz-Carlton", "Best Western", "Holiday Inn"]

    data = []
    for i in range(100 if split != "test" else 200):
        dest = random.choice(cities)
        days = random.randint(3, 10)
        people = random.randint(1, 4)
        budget = random.randint(1000, 5000) * people

        # Generate flight options
        flights = []
        for _ in range(5):
            flights.append({
                "airline": random.choice(airlines),
                "price": random.randint(300, 1200) * people,
                "duration": round(random.uniform(2, 15), 1)
            })

        # Generate hotel options
        hotels_list = []
        for _ in range(5):
            hotels_list.append({
                "name": random.choice(hotels),
                "price_per_night": random.randint(100, 500),
                "rating": round(random.uniform(3.5, 5.0), 1)
            })

        # Compute optimal cost
        best_flight = min(flights, key=lambda x: x["price"])
        best_hotel = min(hotels_list, key=lambda x: x["price_per_night"])
        optimal_cost = best_flight["price"] + best_hotel["price_per_night"] * days

        item = {
            "id": f"{split}_{i:04d}",
            "user_request": f"{dest} for {days} days, budget ${budget}, {people} people",
            "constraints": {
                "destination": dest,
                "days": days,
                "people": people,
                "budget": budget
            },
            "search_results": {
                "flights": flights,
                "hotels": hotels_list
            },
            "ground_truth": {
                "optimal_flight": best_flight,
                "optimal_hotel": best_hotel,
                "optimal_cost": optimal_cost,
                "within_budget": optimal_cost <= budget
            }
        }
        data.append(item)

    with open(data_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return data


DATA_CACHE = {}


class TravelPlanningEnv:
    def __init__(self, split: str = "dev", seed: int = 42):
        self.split = split
        self.seed = seed
        self.current_item = None
        self.itinerary = {"flights": [], "hotels": [], "activities": []}

        if split not in DATA_CACHE:
            DATA_CACHE[split] = load_data(split)
        self.data = DATA_CACHE[split]
        self.rng = random.Random(seed)

    def reset(self):
        self.current_item = self.rng.choice(self.data)
        self.itinerary = {"flights": [], "hotels": [], "activities": []}
        return {
            "user_request": self.current_item["user_request"],
            "constraints": self.current_item["constraints"],
            "search_results": self.current_item["search_results"],
            "current_itinerary": {}
        }

    def step(self, action: dict):
        action_type = action.get("action", "")
        parameters = action.get("parameters", {})

        reward = 0.0
        done = False
        info = {}

        if action_type == "search_flights":
            reward = 0.1
            info["searched"] = "flights"

        elif action_type == "search_hotels":
            reward = 0.1
            info["searched"] = "hotels"

        elif action_type == "add_flight":
            flight_id = parameters.get("flight_id", 0)
            if 0 <= flight_id < len(self.current_item["search_results"]["flights"]):
                flight = self.current_item["search_results"]["flights"][flight_id]
                self.itinerary["flights"].append(flight)
                reward = 0.2

        elif action_type == "add_hotel":
            hotel_id = parameters.get("hotel_id", 0)
            if 0 <= hotel_id < len(self.current_item["search_results"]["hotels"]):
                hotel = self.current_item["search_results"]["hotels"][hotel_id]
                self.itinerary["hotels"].append(hotel)
                reward = 0.2

        elif action_type == "add_activity":
            reward = 0.1

        elif action_type == "finalize":
            # Evaluate final itinerary
            gt = self.current_item["ground_truth"]
            total_cost = sum(f["price"] for f in self.itinerary["flights"])
            total_cost += sum(h["price_per_night"] * self.current_item["constraints"]["days"] for h in self.itinerary["hotels"])

            within_budget = total_cost <= self.current_item["constraints"]["budget"]
            cost_ratio = total_cost / max(gt["optimal_cost"], 1)

            if within_budget and cost_ratio <= 1.2:
                reward = 1.0
            elif within_budget:
                reward = 0.7
            else:
                reward = 0.3

            info["total_cost"] = total_cost
            info["budget"] = self.current_item["constraints"]["budget"]
            info["within_budget"] = within_budget
            info["cost_ratio"] = cost_ratio
            done = True

        else:
            reward = -0.1

        return {
            "observation": {
                "user_request": self.current_item["user_request"],
                "constraints": self.current_item["constraints"],
                "search_results": self.current_item["search_results"],
                "current_itinerary": self.itinerary
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
    with open("/app/tasks/travel_planning/task.json") as f:
        return jsonify(json.load(f))


@app.route("/reset", methods=["POST"])
def reset():
    global _env
    data = request.get_json() or {}
    split = data.get("split", "dev")
    seed = data.get("seed", 42)
    _env = TravelPlanningEnv(split=split, seed=seed)
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