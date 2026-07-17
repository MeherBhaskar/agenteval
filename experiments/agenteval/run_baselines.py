#!/usr/bin/env python3
"""
Run all baseline agents on all tasks (dev split, 3 seeds).
"""

import json
import argparse
from pathlib import Path
from runner import EvaluationRunner
from baselines import BASELINES


def main():
    parser = argparse.ArgumentParser(description="Run baseline agents on AgentEval")
    parser.add_argument("--tasks", default="tasks/core", help="Tasks directory")
    parser.add_argument("--output", default="results/baselines", help="Output directory")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test", "challenge"])
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per task per seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456], help="Random seeds")
    parser.add_argument("--agents", nargs="+", default=list(BASELINES.keys()), help="Agents to run")
    parser.add_argument("--categories", nargs="+", help="Filter by category")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = EvaluationRunner(Path(args.tasks), output_dir, use_mock=True)

    all_results = {}

    for agent_name in args.agents:
        if agent_name not in BASELINES:
            print(f"Unknown agent: {agent_name}")
            continue

        print(f"\n{'='*60}")
        print(f"RUNNING BASELINE: {agent_name}")
        print(f"{'='*60}")

        agent_results = {}

        for seed in args.seeds:
            print(f"\n--- Seed {seed} ---")
            agent = BASELINES[agent_name]()
            
            # Filter tasks by category if specified
            tasks_to_run = []
            for task_id, task in runner.tasks.items():
                if args.categories and task.category not in args.categories:
                    continue
                tasks_to_run.append(task_id)

            for task_id in tasks_to_run:
                print(f"  {task_id}...", end=" ", flush=True)
                try:
                    results = runner.run_task(task_id, agent, args.split, args.episodes, seed)
                    key = f"{task_id}_seed{seed}"
                    agent_results[key] = [r.__dict__ for r in results]
                    print(f"success={results[0].success}, reward={results[0].reward:.2f}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    agent_results[f"{task_id}_seed{seed}"] = {"error": str(e)}

        # Save agent results
        with open(output_dir / f"{agent_name}_results.json", "w") as f:
            json.dump(agent_results, f, indent=2)

        all_results[agent_name] = agent_results
        print(f"\nCompleted {agent_name}: {len(agent_results)} task-seed combos")

    # Generate cross-agent summary
    summary = {}
    for agent_name, results in all_results.items():
        successful = [r for r in results.values() if isinstance(r, list) and r and r[0].get("success", False)]
        total = len(results)
        summary[agent_name] = {
            "success_rate": len(successful) / total if total > 0 else 0,
            "total_runs": total,
            "successful": len(successful),
            "avg_reward": sum(r[0].get("reward", 0) for r in successful) / len(successful) if successful else 0
        }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for agent, stats in summary.items():
        print(f"  {agent:20s}: success={stats['success_rate']:.1%}, avg_reward={stats['avg_reward']:.3f}")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
