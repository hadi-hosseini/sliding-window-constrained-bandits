from functools import lru_cache

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import LLMRouterConfig

config = LLMRouterConfig()
CSV_FILE = config.data_dir / config.csv_file
FEATURE_FILE = config.data_dir / config.feature_file

df = pd.read_csv(CSV_FILE)
feature_matrix = np.load(FEATURE_FILE)
models = sorted(df.model.unique())

K = len(models)
d = feature_matrix.shape[1]
features = {}
reward_table = {}
cost_table = {}

for i, row in tqdm(df.iterrows(), total=len(df), desc="Loading dataset"):
    key = (int(row.query_id), row.model)
    features[key] = feature_matrix[i]
    reward_table[key] = float(row.reward)
    cost_table[key] = float(row.cost)

query_ids = sorted(df.query_id.unique())


def compute_offline_optimal_reward(query_order, T, budget, config):
    window_size = config.window_size
    dp = {tuple(): 0.0}

    for t in range(T):
        q = query_order[t]
        new_dp = {}

        for history, total_reward in dp.items():
            for model in models:
                cost = cost_table[(q, model)]
                reward = reward_table[(q, model)]
                new_history = history + (cost,)

                if len(new_history) >= window_size:
                    window_cost = sum(new_history[-window_size:])

                    if window_cost > budget:
                        continue

                new_history = new_history[-(window_size - 1):]
                new_reward = total_reward + reward

                if (new_history not in new_dp or new_reward > new_dp[new_history]):
                    new_dp[new_history] = new_reward

        dp = new_dp
    return max(dp.values())


class LLMRoutingEnvironment:
    def __init__(self, query_order, budget, config):
        self.query_order = query_order
        self.budget = budget
        self.window_size = config.window_size
        self.history = []
        self.expected_reward = 0.0
        self.violations = 0

    def feasible_actions(self, t):
        q = self.query_order[t]
        recent = self.history[-(self.window_size - 1):]
        used_cost = sum(cost for _, cost in recent)
        actions = []

        for action, model in enumerate(models):
            cost = cost_table[(q, model)]
            if used_cost + cost <= self.budget:
                actions.append(action)

        return actions

    def step(self, t, action):
        q = self.query_order[t]

        if action is None:
            reward = 0.0
            cost = 0.0
            self.history.append((None, cost))
            return reward

        model = models[action]
        reward = reward_table[(q, model)]
        cost = cost_table[(q, model)]
        self.history.append((action, cost))
        self.expected_reward += reward

        if len(self.history) >= self.window_size:
            window_cost = sum(cost for _, cost in self.history[-self.window_size:])
            if window_cost > self.budget:
                self.violations += 1

        return reward


class OptimisticLinearBandit:
    def __init__(self, config, d):
        self.lambda_reg = config.lambda_reg
        self.delta = config.delta
        self.noise_r = config.noise_r
        self.d = d
        self.V = self.lambda_reg * np.eye(d)
        self.b = np.zeros(d)

    def theta_hat(self):
        return np.linalg.solve(self.V, self.b)

    def beta(self):
        return self.noise_r * np.sqrt(self.d * np.log(1.0 / self.delta))

    def score(self, x):
        theta = self.theta_hat()
        uncertainty = np.sqrt(x @ np.linalg.solve(self.V, x))
        return theta @ x + self.beta() * uncertainty

    def logdet(self):
        return np.linalg.slogdet(self.V)[1]

    def update(self, x, y):
        self.V += np.outer(x, x)
        self.b += y * x


class OFULPolicy:
    def __init__(self, config, d, seed=0):
        self.rng = np.random.default_rng(seed)
        self.model = OptimisticLinearBandit(config=config, d=d)
        self.n_updates = 0

    def choose(self, actions, q):
        actions = np.asarray(actions, dtype=int)

        if actions.size == 0:
            return None

        scores = np.array([self.model.score(features[(q, models[action])]) for action in actions])
        max_score = np.max(scores)
        candidates = actions[np.isclose(scores, max_score)]
        self.n_updates += 1
        return int(self.rng.choice(candidates))

    def update(self, x, reward):
        self.model.update(x, reward)




class RareUpdateOptimisticPlanning:
    def __init__(self, config, d, budget, models, features, cost_table, seed=0):
        self.model = OptimisticLinearBandit(config=config, d=d)
        self.config = config
        self.models = models
        self.features = features
        self.cost_table = cost_table
        self.K = len(models)
        self.window_size = config.window_size
        self.det_factor = config.det_factor
        self.planning_horizon = config.planning_horizon
        self.budget = budget
        self.rng = np.random.default_rng(seed)
        self.episode_logdet = None
        self.frozen_scores = {}
        self.frozen_theta = None
        self.frozen_V_inv = None
        self.frozen_beta = None
        self.n_updates = 0

    def start_new_episode(self):
        self.optimistic_value.cache_clear()
        self.episode_logdet = self.model.logdet()
        self.frozen_theta = self.model.theta_hat().copy()
        self.frozen_V_inv = np.linalg.inv(self.model.V).copy()
        self.frozen_beta = self.model.beta()
        self.n_updates += 1


    def frozen_score(self, q, action):
        x = self.features[(q, self.models[action])]
        uncertainty = np.sqrt(x @ self.frozen_V_inv @ x)
        return self.frozen_theta @ x + self.frozen_beta * uncertainty

    def determinant_doubled(self):
        if self.episode_logdet is None:
            return True

        return (self.model.logdet() > self.episode_logdet + np.log(self.det_factor))

    def transition(self, state, cost):
        return tuple((list(state) + [cost])[-(self.window_size - 1):])

    def feasible_actions_from_state(self, state, q):
        used_cost = sum(state)
        feasible = []

        for action in range(self.K):
            cost = self.cost_table[(q, self.models[action])]
            if used_cost + cost <= self.budget:
                feasible.append(action)

        return feasible

    @lru_cache(maxsize=None)
    def optimistic_value(self, state, queries, horizon):
        if horizon <= 0:
            return 0.0

        q = queries[0]
        feasible_actions = self.feasible_actions_from_state(state, q)

        if len(feasible_actions) == 0:
            return 0.0

        best_value = -np.inf
        for action in feasible_actions:
            cost = self.cost_table[(q, self.models[action])]
            next_state = self.transition(state, cost)
            value = (self.frozen_score(q, action) + self.optimistic_value(next_state, queries[1:], horizon - 1))
            best_value = max(best_value, value)

        return best_value

    def planning_action(self, state, queries, horizon):
        q = queries[0]
        feasible_actions = self.feasible_actions_from_state(state, q)

        if len(feasible_actions) == 0:
            return None

        best_value = -np.inf
        candidates = []
        for action in feasible_actions:
            cost = self.cost_table[(q, self.models[action])]
            next_state = self.transition(state, cost)
            value = (self.frozen_score(q, action) + self.optimistic_value(next_state, queries[1:], horizon - 1))

            if value > best_value:
                best_value = value
                candidates = [action]

            elif np.isclose(value, best_value):
                candidates.append(action)

        return int(self.rng.choice(candidates))

    def choose(self, state, queries, t, T):
        queries = tuple(queries)

        if len(queries) == 0:
            return None

        horizon = min(T - t, self.planning_horizon, len(queries))
        planning_queries = queries[:horizon]
        q = planning_queries[0]
        feasible_actions = self.feasible_actions_from_state(state, q)

        if len(feasible_actions) == 0:
            return None

        if self.determinant_doubled():
            self.start_new_episode()

        return self.planning_action(state=state, queries=planning_queries, horizon=horizon)

    def update(self, x, y):
        if x is not None:
            self.model.update(x, y)



class PrimalDualOFUL:

    def __init__(self, budget, config, d, models, features, cost_table, seed=0):
        self.model = OptimisticLinearBandit(config=config, d=d)
        self.budget = budget
        self.window_size = config.window_size
        self.eta = config.dual_eta
        self.models = models
        self.features = features
        self.cost_table = cost_table
        self.K = len(models)
        self.rng = np.random.default_rng(seed)
        self.lambda_dual = 0.0
        self.history = []
        self.n_updates = 0
        self.t = 0

    def choose(self, actions, q):
        actions = np.asarray(actions, dtype=int)

        if actions.size == 0:
            return None

        scores = []
        for action in actions:
            model = self.models[action]
            x = self.features[(q, model)]
            cost = self.cost_table[(q, model)]
            optimistic_reward = self.model.score(x)
            lagrangian_score = optimistic_reward - self.lambda_dual * cost
            scores.append(lagrangian_score)
        scores = np.asarray(scores)

        max_score = scores.max()
        candidates = actions[np.isclose(scores, max_score)]
        self.n_updates += 1

        return int(self.rng.choice(candidates))

    def update(self, x, y, action, cost):
        if action is None:
            cost = 0.0
            self.history.append((None, cost))
        else:
            self.model.update(x, y)
            self.history.append((action, cost))

        recent = self.history[-self.window_size:]

        window_cost = sum(cost for _, cost in recent)
        constraint_residual = window_cost - self.budget
        self.t += 1
        eta_t = self.eta / np.sqrt(self.t)
        self.lambda_dual = max(0.0, self.lambda_dual + eta_t * constraint_residual)


def run_policy(policy_name, T, seed, budget):
    rng = np.random.default_rng(seed)
    query_order = rng.choice(query_ids, size=T, replace=False)
    env = LLMRoutingEnvironment(query_order=query_order, budget=budget, config=config)
    offline_optimal = compute_offline_optimal_reward(query_order=query_order, T=T, budget=budget, config=config)

    if policy_name == "Rare-Update Optimistic Planning":
        agent = RareUpdateOptimisticPlanning(config=config, d=d, budget=budget, models=models, features=features, cost_table=cost_table, seed=seed)

    elif policy_name == "Myopic Feasible OFUL":
        agent = OFULPolicy(config=config, d=d, seed=seed)

    elif policy_name == "Primal-Dual OFUL":
        agent = PrimalDualOFUL(budget=budget, config=config, d=d, models=models, features=features, cost_table=cost_table, seed=seed)

    elif policy_name == "Unconstrained OFUL":
        agent = OFULPolicy(config=config, d=d, seed=seed)

    state = tuple()

    for t in range(T):
        q = query_order[t]

        if policy_name == "Rare-Update Optimistic Planning":
            queries = tuple(query_order[t:min(T, t + config.planning_horizon)])
            action = agent.choose(state=state, queries=queries, t=t, T=T)
        else:
            if policy_name in ("Unconstrained OFUL", "Primal-Dual OFUL"):
                actions = list(range(K))
            else:
                actions = env.feasible_actions(t)

            if len(actions) == 0:
                continue

            action = agent.choose(actions, q)

        reward = env.step(t, action)
        x = None

        if action is not None:
            x = features[(q, models[action])]
            if policy_name == "Primal-Dual OFUL":
                cost = cost_table[(q, models[action])]
                agent.update(x, reward, action, cost)
            else:
                agent.update(x, reward)

        if policy_name == "Rare-Update Optimistic Planning":
            if action is None:
                state = agent.transition(state, 0.0)
            else:
                cost = cost_table[(q, models[action])]
                state = agent.transition(state, cost)

    return {
        "expected_reward": env.expected_reward,
        "average_reward": env.expected_reward / T,
        "offline_optimal": offline_optimal,
        "regret": offline_optimal - env.expected_reward,
        "normalized_regret": (offline_optimal - env.expected_reward) / T,
        "violations": env.violations,
        "updates": agent.n_updates,
    }


def run():
    results = []
    final_file = config.output_dir / "experiment_results_budget.csv"

    for budget in config.budget_values:
        print("\n==============================")
        print("Running budget:", budget)
        print("==============================")

        for T in tqdm(config.t_values, desc=f"Budget {budget}"):
            for run_id in range(config.n_runs):
                seed = config.random_seed + run_id

                for policy in config.policies:
                    result = run_policy(policy_name=policy, T=T, seed=seed, budget=budget)
                    results.append({"budget": budget, "T": T, "run": run_id, "policy": policy, **result})

    results = pd.DataFrame(results)
    results.to_csv(final_file, index=False)

    print("\nFinal saved:", final_file)

    summary = (
        results
        .groupby(["budget", "T", "policy"], as_index=False)
        .agg(
            mean_expected_reward=("expected_reward", "mean"),
            std_expected_reward=("expected_reward", "std"),
            mean_average_reward=("average_reward", "mean"),
            std_average_reward=("average_reward", "std"),
            mean_offline_optimal=("offline_optimal", "mean"),
            std_offline_optimal=("offline_optimal", "std"),
            mean_regret=("regret", "mean"),
            std_regret=("regret", "std"),
            mean_normalized_regret=("normalized_regret", "mean"),
            std_normalized_regret=("normalized_regret", "std"),
            mean_violations=("violations", "mean"),
            std_violations=("violations", "std"),
            mean_updates=("updates", "mean"),
            std_updates=("updates", "std"),
        )
    )

    summary_file = config.output_dir / "experiment_summary_budget.csv"
    summary.to_csv(summary_file, index=False)

    print("\nSummary saved:", summary_file)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    run()