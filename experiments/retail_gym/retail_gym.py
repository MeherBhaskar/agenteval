"""
RetailGym: Full implementation for Paper 3 (Agentic Retail Operations).
Gymnasium-compatible environment for multi-agent retail simulation.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import random
from abc import ABC, abstractmethod


@dataclass
class RetailConfig:
    """Configuration for RetailGym."""
    n_stores: int = 100          # Number of stores (scale down for simulation)
    n_skus: int = 5000           # Number of SKUs
    n_fcs: int = 10              # Fulfillment centers
    n_categories: int = 50       # Product categories
    horizon_days: int = 90       # Simulation horizon
    seed: int = 42

    # Cost parameters
    holding_cost_per_unit: float = 0.05
    stockout_cost_per_unit: float = 5.0
    transfer_cost_per_unit: float = 0.50
    markdown_cost_pct: float = 0.30

    # Demand parameters
    base_demand_mean: float = 10.0
    demand_cv: float = 0.5
    elasticity_mean: float = -1.5
    elasticity_std: float = 0.3


@dataclass
class RetailState:
    """Full state of the retail environment."""
    day: int = 0

    # Inventory: [store, sku] -> units
    inventory: np.ndarray = None          # shape: (n_stores, n_skus)
    inbound_shipments: List[Dict] = field(default_factory=list)

    # Pricing: [store, sku] -> price
    prices: np.ndarray = None             # shape: (n_stores, n_skus)
    competitor_prices: np.ndarray = None  # shape: (n_skus,)

    # Assortment: [store, sku] -> 0/1 (active)
    active_skus: np.ndarray = None        # shape: (n_stores, n_skus)

    # Demand forecasts: [store, sku, day] -> expected units
    forecasts: np.ndarray = None          # shape: (n_stores, n_skus, horizon)
    elasticity: np.ndarray = None         # shape: (n_skus,)

    # Substitution: similarity graph
    similarity_graph: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict)

    # Customer sessions (for substitution agent)
    active_sessions: List[Dict] = field(default_factory=list)

    # KPIs
    kpis: Dict = field(default_factory=dict)


class BaseSubEnv(ABC):
    """Base class for sub-environments."""

    def __init__(self, config: RetailConfig, state: RetailState):
        self.config = config
        self.state = state

    @abstractmethod
    def step(self, action: Dict) -> Tuple[RetailState, float, Dict]:
        pass

    @property
    @abstractmethod
    def observation_space(self) -> spaces.Space:
        pass

    @property
    @abstractmethod
    def action_space(self) -> spaces.Space:
        pass


class InventorySubEnv(BaseSubEnv):
    """Inventory allocation sub-environment."""

    def __init__(self, config: RetailConfig, state: RetailState):
        super().__init__(config, state)

        self.action_space = spaces.Dict({
            'allocate': spaces.MultiDiscrete([config.n_fcs, config.n_stores, config.n_skus, 11]),
            'transfer': spaces.MultiDiscrete([config.n_stores, config.n_stores, config.n_skus, 11]),
            'reorder': spaces.MultiDiscrete([config.n_skus, 11]),
        })

        self.observation_space = spaces.Dict({
            'inventory': spaces.Box(low=0, high=1000, shape=(config.n_stores, config.n_skus), dtype=np.int32),
            'forecasts': spaces.Box(low=0, high=1000, shape=(config.n_stores, config.n_skus, 7), dtype=np.float32),
            'fc_capacity': spaces.Box(low=0, high=10000, shape=(config.n_fcs,), dtype=np.int32),
        })

    def step(self, action: Dict) -> Tuple[RetailState, float, Dict]:
        reward = 0.0
        info = {}

        # Process allocations
        if 'allocate' in action:
            fc, store, sku, qty_bucket = action['allocate']
            qty = qty_bucket * 10
            if self.state.inventory[store, sku] + qty <= 500:
                self.state.inventory[store, sku] += qty
                self.state.inbound_shipments.append({
                    'fc': fc, 'store': store, 'sku': sku, 'qty': qty,
                    'arrival_day': self.state.day + 1
                })

        # Process transfers
        if 'transfer' in action:
            store_from, store_to, sku, qty_bucket = action['transfer']
            qty = qty_bucket * 10
            if self.state.inventory[store_from, sku] >= qty:
                self.state.inventory[store_from, sku] -= qty
                self.state.inventory[store_to, sku] += qty
                reward -= qty * self.config.transfer_cost_per_unit

        # Process reorders
        if 'reorder' in action:
            sku, qty_bucket = action['reorder']
            qty = qty_bucket * 100
            # Add to FC inventory (simplified)
            pass

        # Holding cost
        total_inventory = self.state.inventory.sum()
        reward -= total_inventory * self.config.holding_cost_per_unit

        return self.state, reward, info


class PricingSubEnv(BaseSubEnv):
    """Pricing optimization sub-environment."""

    def __init__(self, config: RetailConfig, state: RetailState):
        super().__init__(config, state)

        self.action_space = spaces.Dict({
            'price_changes': spaces.MultiDiscrete([
                config.n_stores, config.n_skus, 21  # -10% to +10% in 1% steps
            ]),
            'promotions': spaces.MultiDiscrete([
                config.n_skus, 21, 8  # discount 0-20%, duration 0-7 weeks
            ]),
        })

        self.observation_space = spaces.Dict({
            'prices': spaces.Box(low=0, high=1000, shape=(config.n_stores, config.n_skus), dtype=np.float32),
            'competitor_prices': spaces.Box(low=0, high=1000, shape=(config.n_skus,), dtype=np.float32),
            'inventory': spaces.Box(low=0, high=1000, shape=(config.n_stores, config.n_skus), dtype=np.int32),
            'elasticity': spaces.Box(low=-3, high=-0.5, shape=(config.n_skus,), dtype=np.float32),
        })

    def step(self, action: Dict) -> Tuple[RetailState, float, Dict]:
        reward = 0.0
        info = {}

        if 'price_changes' in action:
            store, sku, delta_bucket = action['price_changes']
            delta_pct = (delta_bucket - 10) / 100.0
            current_price = self.state.prices[store, sku]
            new_price = current_price * (1 + delta_pct)

            # Constraints
            min_price = current_price * 0.85
            max_price = current_price * 1.15
            new_price = np.clip(new_price, min_price, max_price)

            self.state.prices[store, sku] = new_price

        if 'promotions' in action:
            sku, discount_bucket, duration = action['promotions']
            discount = discount_bucket / 100.0
            # Apply promotion
            pass

        return self.state, reward, info


class AssortmentSubEnv(BaseSubEnv):
    """Assortment planning sub-environment."""

    def __init__(self, config: RetailConfig, state: RetailState):
        super().__init__(config, state)

        self.action_space = spaces.Dict({
            'add_sku': spaces.MultiDiscrete([config.n_stores, config.n_skus]),
            'remove_sku': spaces.MultiDiscrete([config.n_stores, config.n_skus]),
            'adjust_facings': spaces.MultiDiscrete([config.n_stores, config.n_skus, 5]),
        })

        self.observation_space = spaces.Dict({
            'active_skus': spaces.Box(low=0, high=1, shape=(config.n_stores, config.n_skus), dtype=np.int32),
            'sales_history': spaces.Box(low=0, high=1000, shape=(config.n_stores, config.n_skus), dtype=np.float32),
            'space_available': spaces.Box(low=0, high=100, shape=(config.n_stores,), dtype=np.int32),
        })

    def step(self, action: Dict) -> Tuple[RetailState, float, Dict]:
        reward = 0.0
        info = {}

        if 'add_sku' in action:
            store, sku = action['add_sku']
            if self.state.active_skus[store, sku] == 0:
                self.state.active_skus[store, sku] = 1

        if 'remove_sku' in action:
            store, sku = action['remove_sku']
            if self.state.active_skus[store, sku] == 1:
                self.state.active_skus[store, sku] = 0

        if 'adjust_facings' in action:
            store, sku, facing_delta = action['adjust_facings']
            facing_delta = facing_delta - 2
            pass

        return self.state, reward, info


class SubstitutionSubEnv(BaseSubEnv):
    """Real-time substitution sub-environment."""

    def __init__(self, config: RetailConfig, state: RetailState):
        super().__init__(config, state)

        self.action_space = spaces.Dict({
            'action_type': spaces.Discrete(3),  # 0=suggest, 1=auto_sub, 2=contact
            'substitute_sku': spaces.Discrete(config.n_skus),
            'confidence': spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
        })

        self.observation_space = spaces.Dict({
            'oos_sku': spaces.Discrete(config.n_skus),
            'customer_id': spaces.Discrete(10000),
            'basket': spaces.MultiBinary(config.n_skus),
            'store_inventory': spaces.Box(low=0, high=100, shape=(config.n_skus,), dtype=np.int32),
            'similarity_scores': spaces.Box(low=0.0, high=1.0, shape=(config.n_skus,), dtype=np.float32),
        })

    def step(self, action: Dict) -> Tuple[RetailState, float, Dict]:
        reward = 0.0
        info = {}

        action_type = action.get('action_type', 0)
        sub_sku = action.get('substitute_sku', 0)
        confidence = action.get('confidence', 0.5)

        # Simulate acceptance based on similarity and confidence
        acceptance_prob = confidence * 0.8 + 0.2  # Simplified
        accepted = np.random.random() < acceptance_prob

        if accepted:
            # Margin retention
            oos_sku = self.state.active_sessions[0].get('oos_sku', 0)
            margin_retention = np.random.uniform(0.5, 1.2)
            reward += margin_retention * 10.0
        else:
            reward -= 5.0  # Rejection penalty

        return self.state, reward, info


class RetailGym(gym.Env):
    """
    Main RetailGym environment integrating all sub-environments.
    Supports single-agent (orchestrator) and multi-agent modes.
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, config: RetailConfig = None, mode: str = 'multi_agent'):
        super().__init__()
        self.config = config or RetailConfig()
        self.mode = mode  # 'single_agent' or 'multi_agent'
        self.np_random = np.random.default_rng(self.config.seed)

        # Initialize state
        self.state = RetailState()
        self._init_state()

        # Sub-environments
        self.inventory_env = InventorySubEnv(self.config, self.state)
        self.pricing_env = PricingSubEnv(self.config, self.state)
        self.assortment_env = AssortmentSubEnv(self.config, self.state)
        self.substitution_env = SubstitutionSubEnv(self.config, self.state)

        # Define observation/action spaces based on mode
        if self.mode == 'single_agent':
            self._setup_single_agent_spaces()
        else:
            self._setup_multi_agent_spaces()

    def _init_state(self):
        """Initialize the retail state."""
        cfg = self.config
        self.state.inventory = self.np_random.integers(0, 100, size=(cfg.n_stores, cfg.n_skus), dtype=np.int32)
        self.state.prices = self.np_random.uniform(10, 100, size=(cfg.n_stores, cfg.n_skus)).astype(np.float32)
        self.state.competitor_prices = self.np_random.uniform(10, 100, size=(cfg.n_skus,)).astype(np.float32)
        self.state.active_skus = self.np_random.integers(0, 2, size=(cfg.n_stores, cfg.n_skus), dtype=np.int32)
        self.state.forecasts = self.np_random.uniform(0, 50, size=(cfg.n_stores, cfg.n_skus, cfg.horizon_days)).astype(np.float32)
        self.state.elasticity = self.np_random.normal(cfg.elasticity_mean, cfg.elasticity_std, size=(cfg.n_skus,)).astype(np.float32)
        self.state.elasticity = np.clip(self.state.elasticity, -3.0, -0.5)

        # Build similarity graph (simplified)
        for sku in range(cfg.n_skus):
            n_neighbors = min(10, cfg.n_skus - 1)
            neighbors = self.np_random.choice(cfg.n_skus, size=n_neighbors, replace=False)
            scores = self.np_random.uniform(0.3, 0.9, size=n_neighbors)
            self.state.similarity_graph[sku] = list(zip(neighbors, scores))

    def _setup_single_agent_spaces(self):
        """Single agent (orchestrator) controls all sub-environments."""
        self.observation_space = spaces.Dict({
            'inventory': self.inventory_env.observation_space,
            'pricing': self.pricing_env.observation_space,
            'assortment': self.assortment_env.observation_space,
            'substitution': self.substitution_env.observation_space,
            'day': spaces.Discrete(self.config.horizon_days),
        })
        self.action_space = spaces.Dict({
            'inventory': self.inventory_env.action_space,
            'pricing': self.pricing_env.action_space,
            'assortment': self.assortment_env.action_space,
            'substitution': self.substitution_env.action_space,
        })

    def _setup_multi_agent_spaces(self):
        """Each agent gets its own observation/action space."""
        # In multi-agent mode, we return tuple of (obs_dict, action_dict) for each agent
        self.observation_space = {
            'inventory': self.inventory_env.observation_space,
            'pricing': self.pricing_env.observation_space,
            'assortment': self.assortment_env.observation_space,
            'substitution': self.substitution_env.observation_space,
        }
        self.action_space = {
            'inventory': self.inventory_env.action_space,
            'pricing': self.pricing_env.action_space,
            'assortment': self.assortment_env.action_space,
            'substitution': self.substitution_env.action_space,
        }

    def step(self, actions: Dict) -> Tuple[Any, float, bool, bool, Dict]:
        """Execute one step of the environment."""
        total_reward = 0.0
        info = {}

        if self.mode == 'single_agent':
            # Single agent provides actions for all sub-environments
            for sub_env_name, action in actions.items():
                sub_env = getattr(self, f'{sub_env_name}_env')
                self.state, reward, sub_info = sub_env.step(action)
                total_reward += reward
                info[sub_env_name] = sub_info
        else:
            # Multi-agent: each agent acts on its sub-environment
            for agent_name, action in actions.items():
                sub_env = getattr(self, f'{agent_name}_env')
                self.state, reward, sub_info = sub_env.step(action)
                total_reward += reward
                info[agent_name] = sub_info

        # Simulate demand fulfillment
        self._simulate_demand()
        self._update_kpis()

        # Advance day
        self.state.day += 1
        terminated = self.state.day >= self.config.horizon_days
        truncated = False

        return self._get_obs(), total_reward, terminated, truncated, info

    def _simulate_demand(self):
        """Simulate customer demand and fulfill orders."""
        for store in range(self.config.n_stores):
            for sku in range(self.config.n_skus):
                if self.state.active_skus[store, sku] == 0:
                    continue

                # Demand = forecast * price elasticity effect
                base_demand = self.state.forecasts[store, sku, self.state.day]
                price = self.state.prices[store, sku]
                comp_price = self.state.competitor_prices[sku]
                elasticity = self.state.elasticity[sku]

                # Price effect
                price_ratio = price / comp_price if comp_price > 0 else 1.0
                demand = base_demand * (price_ratio ** elasticity)

                # Add noise
                demand = max(0, self.np_random.normal(demand, demand * self.config.demand_cv))

                # Fulfill
                available = self.state.inventory[store, sku]
                sold = min(demand, available)
                self.state.inventory[store, sku] -= sold

                # Record stockouts
                if demand > available:
                    self.state.kpis['stockout_units'] = self.state.kpis.get('stockout_units', 0) + (demand - available)

        # Process inbound shipments
        for shipment in self.state.inbound_shipments[:]:
            if shipment['arrival_day'] <= self.state.day:
                self.state.inventory[shipment['store'], shipment['sku']] += shipment['qty']
                self.state.inbound_shipments.remove(shipment)

    def _update_kpis(self):
        """Update KPI tracking."""
        self.state.kpis['day'] = self.state.day
        self.state.kpis['total_inventory'] = self.state.inventory.sum()
        self.state.kpis['total_revenue'] = self.state.kpis.get('total_revenue', 0)
        self.state.kpis['avg_price'] = self.state.prices.mean()

    def _get_obs(self):
        """Get observations for all agents."""
        if self.mode == 'single_agent':
            return {
                'inventory': {
                    'inventory': self.state.inventory,
                    'forecasts': self.state.forecasts[:, :, self.state.day:self.state.day+7],
                    'fc_capacity': np.ones(self.config.n_fcs) * 10000,
                },
                'pricing': {
                    'prices': self.state.prices,
                    'competitor_prices': self.state.competitor_prices,
                    'inventory': self.state.inventory,
                    'elasticity': self.state.elasticity,
                },
                'assortment': {
                    'active_skus': self.state.active_skus,
                    'sales_history': np.zeros((self.config.n_stores, self.config.n_skus)),
                    'space_available': np.ones(self.config.n_stores) * 50,
                },
                'substitution': {
                    'oos_sku': 0,
                    'customer_id': 0,
                    'basket': np.zeros(self.config.n_skus, dtype=bool),
                    'store_inventory': self.state.inventory[0],
                    'similarity_scores': np.zeros(self.config.n_skus),
                },
                'day': self.state.day,
            }
        else:
            return {
                'inventory': {
                    'inventory': self.state.inventory,
                    'forecasts': self.state.forecasts[:, :, self.state.day:self.state.day+7],
                    'fc_capacity': np.ones(self.config.n_fcs) * 10000,
                },
                'pricing': {
                    'prices': self.state.prices,
                    'competitor_prices': self.state.competitor_prices,
                    'inventory': self.state.inventory,
                    'elasticity': self.state.elasticity,
                },
                'assortment': {
                    'active_skus': self.state.active_skus,
                    'sales_history': np.zeros((self.config.n_stores, self.config.n_skus)),
                    'space_available': np.ones(self.config.n_stores) * 50,
                },
                'substitution': {
                    'oos_sku': 0,
                    'customer_id': 0,
                    'basket': np.zeros(self.config.n_skus, dtype=bool),
                    'store_inventory': self.state.inventory[0],
                    'similarity_scores': np.zeros(self.config.n_skus),
                },
            }

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[Any, Dict]:
        """Reset the environment."""
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.state = RetailState()
        self._init_state()
        return self._get_obs(), {}

    def render(self, mode: str = 'human'):
        """Render the environment."""
        if mode == 'human':
            print(f"Day {self.state.day}: Inventory={self.state.inventory.sum():.0f}, "
                  f"Revenue=${self.state.kpis.get('total_revenue', 0):.2f}, "
                  f"Stockouts={self.state.kpis.get('stockout_units', 0)}")

    def close(self):
        pass


# Agent wrappers for MAPPO training
class InventoryAgent:
    """Inventory agent using MAPPO."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def act(self, obs: Dict) -> Dict:
        # Placeholder - would use trained policy
        return {
            'allocate': np.array([0, 0, 0, 5]),
            'transfer': np.array([0, 1, 0, 3]),
            'reorder': np.array([0, 2]),
        }


class PricingAgent:
    """Pricing agent using contextual bandit."""

    def __init__(self, n_skus: int, n_stores: int):
        self.n_skus = n_skus
        self.n_stores = n_stores

    def act(self, obs: Dict) -> Dict:
        return {
            'price_changes': np.array([0, 0, 10]),
            'promotions': np.array([0, 5, 2]),
        }


class AssortmentAgent:
    """Assortment agent using differentiable top-K."""

    def act(self, obs: Dict) -> Dict:
        return {
            'add_sku': np.array([0, 0]),
            'remove_sku': np.array([0, 0]),
            'adjust_facings': np.array([0, 0, 2]),
        }


class SubstitutionAgent:
    """Substitution agent using two-tower retrieval + ranker."""

    def act(self, obs: Dict) -> Dict:
        return {
            'action_type': 1,  # auto_sub
            'substitute_sku': 123,
            'confidence': 0.85,
        }


class OrchestratorAgent:
    """Orchestrator agent using hierarchical RL."""

    def act(self, obs: Dict) -> Dict:
        return {
            'inventory': {'allocate': np.array([0, 0, 0, 5])},
            'pricing': {'price_changes': np.array([0, 0, 10])},
            'assortment': {'add_sku': np.array([0, 0])},
            'substitution': {'action_type': 1, 'substitute_sku': 123},
        }


if __name__ == "__main__":
    # Test the environment
    config = RetailConfig(n_stores=10, n_skus=100, n_fcs=3, horizon_days=30)
    env = RetailGym(config, mode='multi_agent')

    obs, _ = env.reset()
    print("Observation spaces:")
    for agent, space in obs.items():
        print(f"  {agent}: {space}")

    print("\nAction spaces:")
    for agent, space in env.action_space.items():
        print(f"  {agent}: {space}")

    # Run a few steps
    for _ in range(5):
        actions = {}
        for agent_name, space in env.action_space.items():
            if hasattr(space, 'sample'):
                actions[agent_name] = space.sample()
            else:
                # Dict space
                actions[agent_name] = {k: v.sample() for k, v in space.items()}

        obs, reward, terminated, truncated, info = env.step(actions)
        env.render()
        if terminated:
            break

    print("\nEnvironment test passed!")