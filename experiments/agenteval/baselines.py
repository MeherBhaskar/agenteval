#!/usr/bin/env python3
"""
Baseline agents for AgentEval benchmark.
"""

import json
import time
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from dataclasses import dataclass

from runner import BaseAgent, EvaluationRunner, EpisodeResult
from pathlib import Path


# ============================================================
# BASELINE AGENTS
# ============================================================

class ReActAgent(BaseAgent):
    """Standard ReAct agent with tool use."""

    def __init__(self, model: str = "gpt-4o-mini", tools: List[str] = None):
        self.model = model
        self.tools = tools or ["search", "calculate", "code"]
        self.history = []

    @property
    def name(self) -> str:
        return f"ReAct-{self.model}"

    def reset(self):
        self.history = []

    def act(self, observation: str) -> str:
        self.history.append({"role": "user", "content": observation})
        # Simplified - real impl would call LLM with tools
        step = len(self.history) // 2
        if step == 0:
            return "search[relevant information for the task]"
        elif step < 3:
            return "search[more specific details]"
        else:
            return "answer[Based on the retrieved information, here is my response...]"


class PlanAndSolveAgent(BaseAgent):
    """Plan-and-Solve: generates plan first, then executes."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.phase = "plan"
        self.plan = []
        self.step_idx = 0

    @property
    def name(self) -> str:
        return f"PlanAndSolve-{self.model}"

    def reset(self):
        self.phase = "plan"
        self.plan = []
        self.step_idx = 0

    def act(self, observation: str) -> str:
        if self.phase == "plan":
            self.phase = "execute"
            self.plan = [
                "Understand the task requirements",
                "Gather necessary information",
                "Execute the plan",
                "Verify the result"
            ]
            return f"PLAN: {'; '.join(self.plan)}"
        
        if self.step_idx < len(self.plan):
            action = f"execute[{self.plan[self.step_idx]}]"
            self.step_idx += 1
            return action
        
        return "answer[Task completed according to plan]"


class ReflexionAgent(BaseAgent):
    """Reflexion: self-critique and retry on failure."""

    def __init__(self, model: str = "gpt-4o-mini", max_reflections: int = 2):
        self.model = model
        self.max_reflections = max_reflections
        self.reflection_count = 0
        self.attempt_history = []

    @property
    def name(self) -> str:
        return f"Reflexion-{self.model}"

    def reset(self):
        self.reflection_count = 0
        self.attempt_history = []

    def act(self, observation: str) -> str:
        if self.reflection_count == 0:
            return "search[initial information gathering]"
        elif self.reflection_count <= self.max_reflections:
            return f"reflect[Attempt {self.reflection_count} failed: need more specific information]; search[refined query]"
        else:
            return "answer[Best effort based on available information]"


class RandomAgent(BaseAgent):
    """Random baseline - picks random actions."""

    def __init__(self, model: str = "random"):
        self.model = model
        self.actions = [
            "search[random query]",
            "calculate[random expression]",
            "code[random code]",
            "answer[random guess]"
        ]

    @property
    def name(self) -> str:
        return "Random"

    def reset(self):
        pass

    def act(self, observation: str) -> str:
        return random.choice(self.actions)


class ChainOfThoughtAgent(BaseAgent):
    """Chain-of-Thought: explicit reasoning before action."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.step = 0

    @property
    def name(self) -> str:
        return f"CoT-{self.model}"

    def reset(self):
        self.step = 0

    def act(self, observation: str) -> str:
        self.step += 1
        if self.step == 1:
            return "think[Let me break down this task step by step...]"
        elif self.step <= 3:
            return f"think[Reasoning step {self.step}: analyzing the information...]; search[relevant info]"
        else:
            return "answer[Based on my reasoning, the answer is...]"


# ============================================================
# AGENT REGISTRY
# ============================================================

BASELINES = {
    "react": lambda: ReActAgent("gpt-4o-mini"),
    "plan_and_solve": lambda: PlanAndSolveAgent("gpt-4o-mini"),
    "reflexion": lambda: ReflexionAgent("gpt-4o-mini"),
    "cot": lambda: ChainOfThoughtAgent("gpt-4o-mini"),
    "random": lambda: RandomAgent(),
}


def get_baseline(name: str) -> BaseAgent:
    """Get baseline agent by name."""
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(BASELINES.keys())}")
    return BASELINES[name]()


if __name__ == "__main__":
    # Quick test
    for name, factory in BASELINES.items():
        agent = factory()
        agent.reset()
        action = agent.act("Test task: What is the capital of France?")
        print(f"{agent.name}: {action[:80]}...")
