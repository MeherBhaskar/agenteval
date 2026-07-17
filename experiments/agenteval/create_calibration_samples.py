#!/usr/bin/env python3
"""
Create calibration samples from mock evaluation runs.
"""

import json
from pathlib import Path

# Sample trajectories for each task type
SAMPLES = [
    # Fact QA - good trajectory
    {
        "sample_id": "cal_001",
        "task_id": "fact_qa_01_multi_hop",
        "trajectory": [
            {"step": 0, "observation": {"question": "What is the capital of France?", "step": 0}, "action": "search[capital of France]"},
            {"step": 1, "observation": {"context": "Paris is the capital of France.", "step": 1}, "action": "answer[Paris]"}
        ],
        "ground_truth": {"answer": "Paris", "metrics": {"exact_match": 1.0, "f1": 1.0, "retrieval_recall@10": 1.0}}
    },
    # Fact QA - partial
    {
        "sample_id": "cal_002",
        "task_id": "fact_qa_01_multi_hop",
        "trajectory": [
            {"step": 0, "observation": {"question": "Who invented the telephone?", "step": 0}, "action": "search[telephone inventor]"},
            {"step": 1, "observation": {"context": "Alexander Graham Bell invented the telephone in 1876.", "step": 1}, "action": "answer[Alexander Graham Bell]"}
        ],
        "ground_truth": {"answer": "Alexander Graham Bell", "metrics": {"exact_match": 1.0, "f1": 1.0, "retrieval_recall@10": 0.8}}
    },
    # Retail substitution - good
    {
        "sample_id": "cal_003",
        "task_id": "retail_01_substitution",
        "trajectory": [
            {"step": 0, "observation": {"oos_sku": "MILK_1L", "customer_id": "C123", "basket": ["MILK_1L", "BREAD"], "store_inventory": {"MILK_2L": 10, "MILK_500ML": 5}, "similarity_graph": {"MILK_1L": [["MILK_2L", 0.9], ["MILK_500ML", 0.7]]}, "customer_preferences": {"organic": False}}, "action": "suggest[MILK_2L]"}
        ],
        "ground_truth": {"substitute": "MILK_2L", "acceptance_prob": 0.85, "margin_retention": 0.92, "metrics": {"acceptance_rate": 0.85, "margin_retention_pct": 0.92}}
    },
    # Code gen - good
    {
        "sample_id": "cal_004",
        "task_id": "code_gen_01_api_usage",
        "trajectory": [
            {"step": 0, "observation": {"task_description": "Fetch data from REST API", "api_docs": "GET /users returns JSON array", "constraints": ["use requests", "handle errors"]}, "action": "code[import requests; response = requests.get('https://api.example.com/users'); response.raise_for_status(); print(response.json())]"}
        ],
        "ground_truth": {"expected_output": "[{'id': 1, 'name': 'Alice'}]", "metrics": {"test_pass_rate": 1.0, "api_correctness": 1.0}}
    },
    # Web shopping - partial
    {
        "sample_id": "cal_005",
        "task_id": "web_shopping_01_product_finding",
        "trajectory": [
            {"step": 0, "observation": {"query": "wireless headphones under $50", "page_html": "<html>...", "url": "https://shop.example.com/search?q=wireless+headphones", "cart": []}, "action": "click[result_1]"},
            {"step": 1, "observation": {"page_html": "<html>Product page...$39.99</html>", "url": "https://shop.example.com/product/123", "cart": []}, "action": "buy"}
        ],
        "ground_truth": {"purchased": True, "price": 39.99, "metrics": {"purchase_success": 1.0, "price_optimality": 0.8, "query_relevance": 0.9}}
    },
    # Travel planning - good
    {
        "sample_id": "cal_006",
        "task_id": "travel_planning_01_multi_constraint",
        "trajectory": [
            {"step": 0, "observation": {"user_request": "Paris 3 days, budget $2000, 2 people", "constraints": {"budget": 2000, "days": 3, "people": 2}, "search_results": {"flights": [{"price": 400, "airline": "Air France"}], "hotels": [{"price": 150, "nights": 3}]}, "current_itinerary": {}}, "action": "add_flight[Air France, $400]"},
            {"step": 1, "observation": {"user_request": "Paris 3 days, budget $2000, 2 people", "constraints": {"budget": 2000, "days": 3, "people": 2}, "search_results": {}, "current_itinerary": {"flight": 400, "hotel": 450}}, "action": "add_hotel[Hotel Central, $150/night]"},
            {"step": 2, "observation": {"current_itinerary": {"flight": 400, "hotel": 450, "total": 850}}, "action": "finalize"}
        ],
        "ground_truth": {"total_cost": 850, "constraints_met": True, "metrics": {"constraint_satisfaction": 1.0, "total_cost_vs_budget": 0.575, "preference_alignment": 0.9}}
    },
]

def main():
    output_path = Path("/home/meher/agentic-ai-research/experiments/agenteval/calibration_samples.jsonl")
    with open(output_path, "w") as f:
        for sample in SAMPLES:
            f.write(json.dumps(sample) + "\n")
    print(f"Created {len(SAMPLES)} calibration samples at {output_path}")

if __name__ == "__main__":
    main()
