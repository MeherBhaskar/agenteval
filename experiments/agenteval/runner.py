#!/usr/bin/env python3
"""
AgentEval: Minimal Viable Benchmark Infrastructure
Single-file runner for evaluating agents on tasks.
"""

import os
import json
import yaml
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod


# ============================================================
# TASK SPECIFICATION
# ============================================================

@dataclass
class TaskSpec:
    """Task definition loaded from YAML."""
    task_id: str
    category: str
    version: str
    description: str
    environment: Dict  # docker image, cpu, memory, timeout
    interface: Dict    # observation_space, action_space
    splits: Dict       # train/dev/test/challenge sizes
    metrics: List[str] # primary metrics
    metadata: Dict     # domain, difficulty, tags

    @classmethod
    def load_all(cls, path: Path) -> List["TaskSpec"]:
        """Load all tasks from a multi-document YAML file."""
        with open(path) as f:
            docs = list(yaml.safe_load_all(f))
        tasks = []
        for doc in docs:
            if doc and 'task_id' in doc:
                tasks.append(cls(**doc))
        return tasks


# ============================================================
# AGENT INTERFACE
# ============================================================

class BaseAgent(ABC):
    """Base class for all agents."""

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """Return action given observation."""
        pass

    @abstractmethod
    def reset(self):
        """Reset agent state for new episode."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent identifier."""
        pass


class ReActAgent(BaseAgent):
    """Simple ReAct agent for baseline."""

    def __init__(self, model: str = "gpt-4o-mini", tools: List[str] = None):
        self.model = model
        self.tools = tools or []
        self.history = []

    @property
    def name(self) -> str:
        return f"ReAct-{self.model}"

    def reset(self):
        self.history = []

    def act(self, observation: str) -> str:
        self.history.append({"role": "user", "content": observation})
        return "I need to think about this..."


# ============================================================
# ENVIRONMENT WRAPPERS
# ============================================================

class DockerEnvironment:
    """Runs task environments in Docker for isolation."""

    def __init__(self, image: str, cpu: str = "2", memory: str = "4g", timeout: int = 300):
        import docker
        self.image = image
        self.cpu = cpu
        self.memory = memory
        self.timeout = timeout
        self.client = docker.from_env()
        self.container = None

    def start(self, task_id: str, env_vars: Dict = None):
        """Start container for task."""
        self.container = self.client.containers.run(
            self.image,
            detach=True,
            cpu_count=int(float(self.cpu)),
            mem_limit=self.memory,
            environment=env_vars or {},
            name=f"agenteval-{task_id}-{int(time.time())}",
            remove=True
        )

    def execute(self, command: str) -> tuple:
        """Execute command in container, return (exit_code, output)."""
        if not self.container:
            raise RuntimeError("Container not started")
        exec_result = self.container.exec_run(command, timeout=self.timeout)
        return exec_result.exit_code, exec_result.output.decode()

    def stop(self):
        if self.container:
            self.container.stop()
            self.container = None


class MockEnvironment:
    """Mock environment for local testing without Docker."""

    def __init__(self, task_spec: TaskSpec):
        self.task_spec = task_spec
        self.step_count = 0
        self.max_steps = 3

    def start(self, task_id: str, env_vars: Dict = None):
        """Initialize mock environment."""
        self.step_count = 0

    def execute(self, command: str) -> tuple:
        """Simulate environment step - returns mock JSON result."""
        self.step_count += 1
        
        # Return mock result based on task type
        if "fact_qa" in self.task_spec.task_id:
            if self.step_count == 1:
                return 0, json.dumps({
                    "observation": {"context": "Retrieved documents about query...", "step": 1},
                    "reward": 0.3,
                    "done": False,
                    "info": {"tokens": 100, "cost": 0.001}
                })
            else:
                return 0, json.dumps({
                    "observation": {"answer": "Paris", "step": self.step_count},
                    "reward": 1.0,
                    "done": True,
                    "info": {"tokens": 50, "cost": 0.0005}
                })
        elif "retail" in self.task_spec.task_id:
            return 0, json.dumps({
                "observation": {"substitute": "SKU123", "confidence": 0.8},
                "reward": 0.8,
                "done": True,
                "info": {"tokens": 200, "cost": 0.002}
            })
        elif "code_gen" in self.task_spec.task_id:
            return 0, json.dumps({
                "observation": {"test_result": "passed", "output": "Hello World"},
                "reward": 1.0,
                "done": True,
                "info": {"tokens": 500, "cost": 0.005}
            })
        elif "web_shopping" in self.task_spec.task_id:
            return 0, json.dumps({
                "observation": {"page": "product", "price": 29.99},
                "reward": 0.7,
                "done": True,
                "info": {"tokens": 300, "cost": 0.003}
            })
        elif "travel" in self.task_spec.task_id:
            return 0, json.dumps({
                "observation": {"itinerary": "complete", "cost": 1200},
                "reward": 0.9,
                "done": True,
                "info": {"tokens": 400, "cost": 0.004}
            })
        
        # Default
        return 0, json.dumps({
            "observation": {"step": self.step_count},
            "reward": 0.5,
            "done": self.step_count >= self.max_steps,
            "info": {"tokens": 100, "cost": 0.001}
        })

    def stop(self):
        pass


# ============================================================
# EVALUATION RUNNER
# ============================================================

@dataclass
class EpisodeResult:
    task_id: str
    agent_name: str
    episode_id: int
    success: bool
    reward: float
    metrics: Dict[str, float]
    trajectory: List[Dict]
    wall_time: float
    tokens_used: int
    cost_usd: float
    error: Optional[str] = None


class EvaluationRunner:
    """Main evaluation loop."""

    def __init__(self, tasks_dir: Path, output_dir: Path, use_mock: bool = True):
        self.tasks_dir = tasks_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_mock = use_mock

        self.tasks = self._load_tasks()
        self.results = []

    def _load_tasks(self) -> Dict[str, TaskSpec]:
        tasks = {}
        for yaml_file in self.tasks_dir.rglob("*.yaml"):
            try:
                for spec in TaskSpec.load_all(yaml_file):
                    tasks[spec.task_id] = spec
            except Exception as e:
                print(f"Failed to load {yaml_file}: {e}")
        return tasks

    def run_task(self, task_id: str, agent: BaseAgent, split: str = "dev",
                 n_episodes: int = 1, seed: int = 42) -> List[EpisodeResult]:
        """Run agent on task for n_episodes."""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        # Build environment
        if self.use_mock:
            env = MockEnvironment(task)
        else:
            env = DockerEnvironment(
                task.environment.get("image", "agenteval/base:latest"),
                task.environment.get("cpu", "2"),
                task.environment.get("memory", "4g"),
                task.environment.get("timeout", 300)
            )

        results = []

        for ep in range(n_episodes):
            print(f"  Episode {ep+1}/{n_episodes}...")
            agent.reset()
            env.start(task_id, env_vars={"SEED": str(seed + ep), "SPLIT": split})

            try:
                result = self._run_episode(env, agent, task, ep)
                results.append(result)
                self._save_result(result)
            except Exception as e:
                result = EpisodeResult(
                    task_id=task_id,
                    agent_name=agent.name,
                    episode_id=ep,
                    success=False,
                    reward=0.0,
                    metrics={},
                    trajectory=[],
                    wall_time=0.0,
                    tokens_used=0,
                    cost_usd=0.0,
                    error=str(e)
                )
                results.append(result)
                self._save_result(result)
            finally:
                env.stop()

        return results

    def _run_episode(self, env, agent: BaseAgent,
                     task: TaskSpec, episode_id: int) -> EpisodeResult:
        """Single episode rollout."""
        start_time = time.time()
        trajectory = []
        total_reward = 0.0
        tokens = 0
        cost = 0.0

        # Initial observation
        observation = {"task": task.description, "step": 0}

        for step in range(10):  # max 10 steps
            action = agent.act(json.dumps(observation))
            trajectory.append({"step": step, "observation": observation, "action": action})

            # Execute in environment
            exit_code, output = env.execute(json.dumps({"action": action}))

            try:
                result = json.loads(output)
                observation = result.get("observation", {})
                reward = result.get("reward", 0.0)
                done = result.get("done", False)
                info = result.get("info", {})

                total_reward += reward
                tokens += info.get("tokens", 0)
                cost += info.get("cost", 0.0)

                if done:
                    break
            except json.JSONDecodeError:
                observation = {"error": "Invalid environment output", "output": output}

        wall_time = time.time() - start_time
        success = total_reward > 0

        return EpisodeResult(
            task_id=task.task_id,
            agent_name=agent.name,
            episode_id=episode_id,
            success=success,
            reward=total_reward,
            metrics={"total_reward": total_reward},
            trajectory=trajectory,
            wall_time=wall_time,
            tokens_used=tokens,
            cost_usd=cost
        )

    def _save_result(self, result: EpisodeResult):
        """Append result to JSONL."""
        out_file = self.output_dir / "results.jsonl"
        with open(out_file, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

    def run_agent_on_all(self, agent: BaseAgent, split: str = "dev",
                         categories: List[str] = None, n_episodes: int = 1) -> List[EpisodeResult]:
        """Run agent on all tasks (filtered by category)."""
        all_results = []

        for task_id, task in self.tasks.items():
            if categories and task.category not in categories:
                continue

            print(f"\nRunning {agent.name} on {task_id} ({task.category})...")
            results = self.run_task(task_id, agent, split, n_episodes)
            all_results.extend(results)

        self._generate_summary(all_results, agent.name)
        return all_results

    def _generate_summary(self, results: List[EpisodeResult], agent_name: str):
        """Print and save summary statistics."""
        if not results:
            return

        by_task = {}
        for r in results:
            if r.task_id not in by_task:
                by_task[r.task_id] = []
            by_task[r.task_id].append(r)

        print(f"\n{'='*60}")
        print(f"SUMMARY: {agent_name}")
        print(f"{'='*60}")

        for task_id, task_results in by_task.items():
            success_rate = sum(1 for r in task_results if r.success) / len(task_results)
            avg_reward = sum(r.reward for r in task_results) / len(task_results)
            avg_time = sum(r.wall_time for r in task_results) / len(task_results)
            total_cost = sum(r.cost_usd for r in task_results)

            print(f"  {task_id}: success={success_rate:.2%}, reward={avg_reward:.3f}, "
                  f"time={avg_time:.1f}s, cost=${total_cost:.4f}")

        overall_success = sum(1 for r in results if r.success) / len(results)
        overall_reward = sum(r.reward for r in results) / len(results)
        total_cost = sum(r.cost_usd for r in results)

        print(f"\n  OVERALL: success={overall_success:.2%}, reward={overall_reward:.3f}, "
              f"cost=${total_cost:.4f}")

        # Save summary
        summary = {
            "agent": agent_name,
            "n_episodes": len(results),
            "overall_success_rate": overall_success,
            "overall_avg_reward": overall_reward,
            "total_cost_usd": total_cost,
            "by_task": {
                tid: {
                    "success_rate": sum(1 for r in tr if r.success) / len(tr),
                    "avg_reward": sum(r.reward for r in tr) / len(tr),
                    "avg_time": sum(r.wall_time for r in tr) / len(tr),
                    "total_cost": sum(r.cost_usd for r in tr)
                }
                for tid, tr in by_task.items()
            }
        }

        with open(self.output_dir / f"summary_{agent_name}.json", "w") as f:
            json.dump(summary, f, indent=2)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AgentEval Runner")
    parser.add_argument("--tasks", default="tasks", help="Tasks directory")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--agent", default="react", help="Agent type: react")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model for agent")
    parser.add_argument("--task", help="Specific task ID (default: all)")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test", "challenge"])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock", action="store_true", default=True, help="Use mock environment (default)")
    parser.add_argument("--docker", action="store_true", help="Use Docker environments")

    args = parser.parse_args()

    tasks_dir = Path(args.tasks)
    output_dir = Path(args.output)

    # Initialize runner
    use_mock = not args.docker
    runner = EvaluationRunner(tasks_dir, output_dir, use_mock=use_mock)

    # Create agent
    if args.agent == "react":
        agent = ReActAgent(model=args.model)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # Run
    if args.task:
        runner.run_task(args.task, agent, args.split, args.episodes, args.seed)
    else:
        categories = [args.category] if args.category else None
        runner.run_agent_on_all(agent, args.split, categories, args.episodes)


if __name__ == "__main__":
    main()
