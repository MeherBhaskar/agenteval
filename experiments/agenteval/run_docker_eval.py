#!/usr/bin/env python3
"""
Run AgentEval baselines using Docker containers (real task servers).
Runs 5 agents × 5 core tasks × 3 seeds = 75 evaluations per split.
"""
import os
import json
import time
import subprocess
import docker
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import random

# Core task IDs (the 5 main ones)
CORE_TASKS = [
    "fact_qa_01_multi_hop",
    "retail_01_substitution",
    "code_gen_01_api_usage",
    "web_shopping_01_product_finding",
    "travel_planning_01_multi_constraint"
]

# Baseline agents
BASELINE_AGENTS = [
    "react",
    "plan_and_solve",
    "reflexion",
    "cot",
    "random"
]

# Task to Docker image mapping
TASK_IMAGES = {
    "fact_qa_01_multi_hop": "agenteval/fact-qa:v1.0",
    "retail_01_substitution": "agenteval/retail-sim:v1.0",
    "code_gen_01_api_usage": "agenteval/code-gen:v1.0",
    "web_shopping_01_product_finding": "agenteval/web-shopping:v1.0",
    "travel_planning_01_multi_constraint": "agenteval/travel-planning:v1.0"
}

SEEDS = [42, 123, 456]
SPLITS = ["dev", "test"]

@dataclass
class EpisodeResult:
    task_id: str
    agent_name: str
    episode_id: int
    split: str
    seed: int
    success: bool
    reward: float
    metrics: Dict[str, float]
    trajectory: List[Dict]
    wall_time: float
    tokens_used: int
    cost_usd: float
    error: str = None


class DockerEnvironment:
    """Manages a Docker container for a task environment."""

    def __init__(self, task_id: str, seed: int, split: str = "dev"):
        self.task_id = task_id
        self.seed = seed
        self.split = split
        self.image = self._get_image(task_id)
        self.container = None
        self.port = None
        self.base_url = None
        self.client = docker.from_env()

    def _get_image(self, task_id: str) -> str:
        return {
            "fact_qa_01_multi_hop": "agenteval/fact-qa:v1.0",
            "retail_01_substitution": "agenteval/retail-sim:v1.0",
            "code_gen_01_api_usage": "agenteval/code-gen:v1.0",
            "web_shopping_01_product_finding": "agenteval/web-shopping:v1.0",
            "travel_planning_01_multi_constraint": "agenteval/travel-planning:v1.0"
        }.get(task_id, "agenteval/base:latest")

    def start(self):
        """Start the Docker container."""
        # Find free port
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            port = s.getsockname()[1]

        self.port = port
        self.container = self.client.containers.run(
            self.image,
            detach=True,
            ports={'8000/tcp': port},
            environment={
                'SEED': str(self.seed),
                'SPLIT': self.split,
                'TASK_DATA_DIR': '/app/tasks/data'
            },
            cpu_count=2,
            mem_limit='4g',
            remove=True,
            name=f"agenteval-{self.task_id}-{self.seed}-{int(time.time())}"
        )

        # Wait for container to be ready
        self.base_url = f"http://localhost:{port}"
        for _ in range(30):
            try:
                resp = requests.get(f"http://localhost:{port}/health", timeout=2)
                if resp.status_code == 200:
                    break
            except:
                time.sleep(1)
        else:
            raise RuntimeError(f"Container for {self.task_id} failed to start")

        return self.base_url

    def stop(self):
        if self.container:
            try:
                self.container.stop(timeout=10)
            except:
                pass

    def reset(self, seed: int = None):
        """Reset environment."""
        if seed is not None:
            self.seed = seed
        resp = requests.post(f"{self.base_url}/reset", json={"seed": self.seed, "split": self.split}, timeout=30)
        return resp.json()

    def step(self, action: dict):
        resp = requests.post(f"{self.base_url}/step", json=action, timeout=30)
        return resp.json()

    def get_task_spec(self):
        resp = requests.get(f"{self.base_url}/task", timeout=10)
        return resp.json()


class BaseAgent:
    """Base class for agents."""

    def __init__(self, name: str):
        self.name = name
        self.history = []

    def reset(self):
        self.history = []

    def act(self, observation: dict) -> dict:
        raise NotImplementedError


class ReActAgent(BaseAgent):
    def __init__(self):
        super().__init__("ReAct")

    def act(self, observation: dict) -> dict:
        step = len(self.history)
        if step == 0:
            return {"action_type": "search", "content": "relevant information for the task"}
        elif step < 3:
            return {"action_type": "search", "content": "more specific details"}
        else:
            return {"action_type": "answer", "content": "Based on the retrieved information, here is my response..."}


class PlanAndSolveAgent(BaseAgent):
    def __init__(self):
        super().__init__("PlanAndSolve")
        self.phase = "plan"

    def reset(self):
        super().reset()
        self.phase = "plan"

    def act(self, observation: dict) -> dict:
        if self.phase == "plan":
            self.phase = "execute"
            return {"action_type": "plan", "content": "Understand task; gather info; execute; verify"}
        return {"action_type": "execute", "content": "executing step..."}


class ReflexionAgent(BaseAgent):
    def __init__(self):
        super().__init__("Reflexion")
        self.reflection_count = 0

    def reset(self):
        super().reset()
        self.reflection_count = 0

    def act(self, observation: dict) -> dict:
        if self.reflection_count == 0:
            return {"action_type": "search", "content": "initial information gathering"}
        elif self.reflection_count < 2:
            self.reflection_count += 1
            return {"action_type": "reflect", "content": f"Attempt {self.reflection_count} failed; refining approach"}
        return {"action_type": "answer", "content": "Best effort based on available information"}


class CoTAgent(BaseAgent):
    def __init__(self):
        super().__init__("CoT")
        self.step = 0

    def reset(self):
        super().reset()
        self.step = 0

    def act(self, observation: dict) -> dict:
        self.step += 1
        if self.step == 1:
            return {"action_type": "think", "content": "Let me break down this task step by step..."}
        elif self.step <= 3:
            return {"action_type": "think", "content": f"Reasoning step {self.step}: analyzing the information..."}
        return {"action_type": "answer", "content": "Based on my reasoning, the answer is..."}


class RandomAgent(BaseAgent):
    def __init__(self):
        super().__init__("Random")
        self.actions = [
            {"action_type": "search", "content": "random query"},
            {"action_type": "calculate", "content": "random expression"},
            {"action_type": "code", "content": "random code"},
            {"action_type": "answer", "content": "random guess"}
        ]

    def act(self, observation: dict) -> dict:
        return random.choice(self.actions)


AGENTS = {
    "react": lambda: ReActAgent(),
    "plan_and_solve": lambda: PlanAndSolveAgent(),
    "reflexion": lambda: ReflexionAgent(),
    "cot": lambda: CoTAgent(),
    "random": lambda: RandomAgent()
}


def run_episode(env: DockerEnvironment, agent: BaseAgent, episode_id: int, max_steps: int = 10) -> EpisodeResult:
    """Run a single episode."""
    start_time = time.time()
    agent.reset()
    env.reset()

    trajectory = []
    total_reward = 0.0
    tokens = 0
    cost = 0.0
    success = False

    for step in range(max_steps):
        # Get agent action
        obs = env.get_task_spec() if step == 0 else {"status": "running"}
        action = agent.act(obs)
        agent.history.append({"step": step, "observation": obs, "action": action})

        # Execute in environment
        result = env.step(action)

        trajectory.append({
            "step": step,
            "observation": obs,
            "action": action,
            "result": result
        })

        reward = result.get("reward", 0.0)
        total_reward += reward

        if result.get("done", False):
            success = reward > 0.5
            break

    wall_time = time.time() - start_time

    return EpisodeResult(
        task_id=env.task_id,
        agent_name=agent.name,
        episode_id=episode_id,
        split=env.split,
        seed=env.seed,
        success=success,
        reward=total_reward,
        metrics={"total_reward": total_reward},
        trajectory=trajectory,
        wall_time=wall_time,
        tokens_used=tokens,
        cost_usd=cost
    )


def run_evaluation(task_id: str, agent_name: str, split: str, seeds: List[int], episodes_per_seed: int = 1) -> List[EpisodeResult]:
    """Run evaluation for one task-agent-split combination."""
    results = []

    for seed in seeds:
        # Start environment
        env = DockerEnvironment(task_id, seed, split)
        try:
            env.start()
            agent_factory = AGENTS[agent_name]

            for ep in range(episodes_per_seed):
                agent = agent_factory()
                result = run_episode(env, agent, ep)
                results.append(result)
                print(f"  {task_id}/{agent_name}/{split}/seed{seed}/ep{ep}: success={result.success}, reward={result.reward:.3f}")

        finally:
            env.stop()

    return results


def main():
    parser = argparse.ArgumentParser(description="Run AgentEval baselines with Docker")
    parser.add_argument("--tasks", nargs="+", default=CORE_TASKS)
    parser.add_argument("--agents", nargs="+", default=list(AGENTS.keys()))
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output", default="results/docker_eval_results.jsonl")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    all_results = []

    for task_id in args.tasks:
        if task_id not in CORE_TASKS:
            print(f"Skipping unknown task: {task_id}")
            continue

        for agent_name in args.agents:
            if agent_name not in AGENTS:
                print(f"Skipping unknown agent: {agent_name}")
                continue

            for split in args.splits:
                print(f"\n=== {task_id} / {agent_name} / {split} ===")
                results = run_evaluation(task_id, agent_name, split, args.seeds, args.episodes)
                all_results.extend(results)

                # Save incrementally
                with open(args.output, "a") as f:
                    for r in results:
                        f.write(json.dumps(asdict(r)) + "\n")

    # Generate summary
    summary = {}
    for r in all_results:
        key = (r.task_id, r.agent_name, r.split)
        if key not in summary:
            summary[key] = {"success": 0, "total": 0, "rewards": []}
        summary[key]["total"] += 1
        summary[key]["rewards"].append(r.reward)
        if r.success:
            summary[key]["success"] += 1

    print("\n=== SUMMARY ===")
    for (task_id, agent_name, split), stats in summary.items():
        success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        avg_reward = sum(stats["rewards"]) / len(stats["rewards"]) if stats["rewards"] else 0
        print(f"  {task_id} / {agent_name} / {split}: success={success_rate:.1%}, avg_reward={avg_reward:.3f}")

    # Save summary
    with open(args.output.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump({f"{t}/{a}/{s}": v for (t, a, s), v in summary.items()}, f, indent=2)


if __name__ == "__main__":
    import argparse
    import socket
    import requests
    import time
    main()