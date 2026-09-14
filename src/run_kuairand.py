from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import KuaiRandConfig


kuairand_config = KuaiRandConfig()

top_categories = pd.read_csv(kuairand_config.data_dir / "top_categories.csv")
preferences = pd.read_csv(kuairand_config.data_dir / "user_category_preferences.csv")
coverage = pd.read_csv(kuairand_config.data_dir / "user_coverage.csv")

K = len(top_categories)
theta_norm_bound = np.sqrt(K)


def select_eligible_users(coverage, K, config):
    eligible = coverage[(coverage["num_categories"] == K) & (coverage["min_category_interactions"] >= config.min_category_interactions)]
    user_ids = eligible["user_id"].to_numpy()
    rng = np.random.default_rng(config.random_seed)
    user_ids = rng.choice(user_ids, size = config.n_users, replace = False)
    return user_ids


def get_theta(user_id, preferences, config):
    user_data = preferences[preferences["user_id"] == user_id].sort_values("action_id")
    theta_star = user_data[config.theta_column].astype(float).to_numpy()
    return theta_star


def offline_value(theta_star, T, config):
    w = config.window_size
    c = config.max_exposure
    number_of_categories = w // c
    best_categories = np.sort(theta_star)[::-1][:number_of_categories]
    value_per_window = c * best_categories.sum()
    return (T // w) * value_per_window

class RollingExposureEnvironment:

    def __init__(self, theta_star, config, seed=0):
        self.theta = np.asarray(theta_star, dtype=float)
        self.K = len(self.theta)
        self.w = config.window_size
        self.c = config.max_exposure
        self.rng = np.random.default_rng(seed)

        self.history = []
        self.realized_reward = 0
        self.expected_reward = 0.0
        self.n_violations = 0

    def feasible_actions(self):
        recent_history = self.history[-(self.w - 1):]
        counts = Counter(recent_history)
        feasible = [action for action in range(self.K) if counts[action] < self.c]
        return feasible

    def step(self, action):
        if action is None:
            return 0, 0.0, False
        
        action = int(action)
        expected_reward = float(self.theta[action])
        reward = int(self.rng.binomial(1, expected_reward))
        self.history.append(action)
        violation = False

        if len(self.history) >= self.w:
            window = self.history[-self.w:]
            counts = Counter(window)
            violation = any(count > self.c for count in counts.values())

            if violation:
                self.n_violations += 1

        self.realized_reward += reward
        self.expected_reward += expected_reward

        return reward, expected_reward, violation


class OptimisticLinearBandit:

    def __init__(self, K, config, theta_norm_bound):
        self.K = K
        self.lambda_reg = config.lambda_reg
        self.delta = config.delta
        self.noise_r = config.noise_r
        self.theta_norm_bound = theta_norm_bound
        self.v_diag = np.full(K, self.lambda_reg, dtype=float)
        self.b = np.zeros(K, dtype=float)

    def theta_hat(self):
        return self.b / self.v_diag

    def logdet(self):
        return float(np.log(self.v_diag).sum())

    def beta(self):
        logdet_v = self.logdet()
        logdet_lambda = self.K * np.log(self.lambda_reg)
        log_term = 0.5 * (logdet_v - logdet_lambda) + np.log(1.0 / self.delta)
        beta =  self.noise_r * np.sqrt(2.0 * max(log_term, 0.0)) + np.sqrt(self.lambda_reg) * self.theta_norm_bound
        return float(beta)

    def optimistic_scores(self):
        theta_hat = self.theta_hat()
        beta = self.beta()
        bonus = beta / np.sqrt(self.v_diag)
        scores = theta_hat + bonus
        return scores

    def update(self, action, reward):
        action = int(action)
        self.v_diag[action] += 1.0
        self.b[action] += reward


class RareUpdateOptimisticPlanning:

    def __init__(self, K, config, theta_norm_bound, seed=0):
        self.K = K
        self.w = config.window_size
        self.c = config.max_exposure
        self.det_factor = config.det_factor
        self.rng = np.random.default_rng(seed)
        self.model = OptimisticLinearBandit(K=K, config=config, theta_norm_bound=theta_norm_bound)
        self.episode_logdet = None
        self.frozen_scores = None
        self.n_updates = 0
        self.initial_order = self.rng.permutation(K)
        self.planning_horizon = config.planning_horizon

    def start_new_episode(self):
        self.episode_logdet = self.model.logdet()
        self.frozen_scores = self.model.optimistic_scores()
        self.n_updates += 1

    def determinant_doubled(self):
        if self.episode_logdet is None:
            return True

        return self.model.logdet() > self.episode_logdet + np.log(self.det_factor)

    def initialization_action(self, t, feasible_actions):
        index = (t - 1) % self.K
        preferred = int(self.initial_order[index])

        if preferred in feasible_actions:
            return preferred

        return int(self.rng.choice(feasible_actions))


    def transition(self, state, action):
        return tuple(list(state[1:]) + [action])

    def feasible_actions_from_state(self, state):
        counts = Counter(state)
        return [action for action in range(self.K) if counts[action] < self.c]


    @lru_cache(maxsize=None)
    def optimistic_value(self, state, horizon):
        if horizon == 0:
            return 0.0

        best_value = -np.inf
        feasible = self.feasible_actions_from_state(state)

        for action in feasible:
            next_state = self.transition(state, action)
            value = self.frozen_scores[action] + self.optimistic_value(next_state, horizon - 1)
            best_value = max(best_value, value)
        return best_value


    def planning_action(self, state, horizon):
        best_action = None
        best_value = -np.inf
        feasible = self.feasible_actions_from_state(state)

        for action in feasible:
            next_state = self.transition(state, action)

            if horizon == 0:
                value = self.frozen_scores[action]
            else:
                value = self.frozen_scores[action] + self.optimistic_value(next_state, horizon - 1)

            if value > best_value:
                best_value = value
                best_action = action

        return int(best_action)

    def choose(self, state, t, T):
        feasible_actions = self.feasible_actions_from_state(state)

        if len(feasible_actions)==0:
            return None

        if t < self.w:
            return self.initialization_action(t, feasible_actions)

        if self.determinant_doubled():
            self.start_new_episode()
            self.optimistic_value.cache_clear()

        horizon = min(T - t, self.planning_horizon)
        return self.planning_action(state, horizon)


    def update(self, action, reward):
        self.model.update(action, reward)


class OFULPolicy:

    def __init__(self, K, config, theta_norm_bound=None, seed=0):
        self.K = K
        self.rng = np.random.default_rng(seed)
        self.model = OptimisticLinearBandit(K=K, config=config, theta_norm_bound=theta_norm_bound)
        self.n_updates = 0

    def choose(self, actions):
        actions = np.asarray(actions, dtype=int)

        if actions.size == 0:
            return None

        scores = self.model.optimistic_scores()
        action_scores = scores[actions]
        max_score = np.max(action_scores)
        candidates = actions[np.isclose(action_scores, max_score)]
        self.n_updates += 1

        return int(self.rng.choice(candidates))

    def update(self, action, reward):
        self.model.update(action, reward)



class PrimalDualOFUL:
    def __init__(self, K, config, theta_norm_bound=None, seed=0):
        self.K = K
        self.w = config.window_size
        self.c = config.max_exposure
        self.eta = config.dual_eta
        self.rng = np.random.default_rng(seed)
        self.model = OptimisticLinearBandit(K=K, config=config, theta_norm_bound=theta_norm_bound)
        self.lambda_dual = 0.0
        self.history = []
        self.n_updates = 0
        self.t = 0

    def choose(self, actions):
        actions = np.asarray(actions, dtype=int)

        if actions.size == 0:
            return None

        scores = self.model.optimistic_scores()
        recent_history = self.history[-(self.w - 1):]
        counts = Counter(recent_history)
        lagrangian_scores = np.empty(len(actions), dtype=float)

        for i, action in enumerate(actions):
            action = int(action)
            current_count = counts.get(action, 0) + 1
            predicted_violation = max(0, current_count - self.c)
            lagrangian_scores[i] = scores[action] - self.lambda_dual * predicted_violation

        max_score = np.max(lagrangian_scores)
        candidates = actions[np.isclose(lagrangian_scores, max_score)]
        self.n_updates += 1
        return int(self.rng.choice(candidates))

    def update(self, action, reward):
        if action is None:
            return

        action = int(action)
        self.model.update(action, reward)
        self.history.append(action)

        if len(self.history) >= self.w:
            window = self.history[-self.w:]
            counts = Counter(window)
            max_count = max(counts.values())
            violation = max(0, max_count - self.c)
            self.t += 1
            eta_t = self.eta / np.sqrt(self.t)
            self.lambda_dual = max(0.0, self.lambda_dual + eta_t * violation)


def run_policy(theta_star, policy_name, T, seed, config):
    env = RollingExposureEnvironment(theta_star=theta_star, config=config, seed=seed)

    if policy_name == "Rare-Update Optimistic Planning":
        agent = RareUpdateOptimisticPlanning(K=len(theta_star), config=config, theta_norm_bound=theta_norm_bound, seed=seed + 1000)


    elif policy_name == "Myopic Feasible OFUL":
        agent = OFULPolicy(K=len(theta_star), config=config, theta_norm_bound=theta_norm_bound, seed=seed + 1000)

    elif policy_name == "Primal-Dual OFUL":
        agent = PrimalDualOFUL(K=len(theta_star), config=config, theta_norm_bound=theta_norm_bound, seed=seed + 1000)

    elif policy_name == "Unconstrained OFUL":
        agent = OFULPolicy(K=len(theta_star), config=config, theta_norm_bound=theta_norm_bound, seed=seed + 1000)


    for t in range(1, T + 1):

        if policy_name == "Rare-Update Optimistic Planning":
            state = tuple(env.history[-(config.window_size - 1):])
            action = agent.choose(state=state, t=t, T=T)

        elif policy_name == "Primal-Dual OFUL":
            action = agent.choose(np.arange(len(theta_star)))

        elif policy_name == "Myopic Feasible OFUL":
            feasible = (env.feasible_actions())
            action = agent.choose(feasible)

        elif policy_name == "Unconstrained OFUL":
            action = agent.choose(np.arange(len(theta_star)))

        reward, _, _ = env.step(action)
        agent.update(action, reward)

    V_off = offline_value(theta_star=theta_star, T=T, config=config)
    pseudo_regret = (V_off - env.expected_reward)

    return {
        "realized_reward": env.realized_reward,
        "average_realized_reward": env.realized_reward / T,
        "expected_reward": env.expected_reward,
        "average_expected_reward": env.expected_reward / T,
        "offline_value": V_off,
        "pseudo_regret": pseudo_regret,
        "normalized_pseudo_regret": pseudo_regret / T,
        "violations": env.n_violations,
        "updates": agent.n_updates,
    }


def run():
    results = []
    selected_user_ids = select_eligible_users(coverage, K, kuairand_config)

    for user_index, user_id in enumerate(tqdm(selected_user_ids, desc="Processing users", unit="user")):
        theta_star = get_theta(user_id, preferences, kuairand_config)

        for T in kuairand_config.t_values:
            for run_idx in range(kuairand_config.n_runs):
                for policy_name in kuairand_config.policies:
                    seed = kuairand_config.random_seed + 100000 * user_index + 100 * run_idx
                    outcome = run_policy(theta_star=theta_star, policy_name=policy_name, T=T, seed=seed, config=kuairand_config)
                    results.append({"user_id": user_id, "run_idx": run_idx, "T": T, "policy": policy_name, **outcome})

    results_df = pd.DataFrame(results)
    results_df["average_offline_reward"] = results_df["offline_value"] / results_df["T"]

    results_file = kuairand_config.output_dir / "experiment_results.csv"
    results_df.to_csv(results_file, index=False)

    user_results = (
        results_df
        .groupby(["user_id", "T", "policy"], as_index=False)
        .agg(
            offline_value=("offline_value", "mean"),
            average_offline_reward=("average_offline_reward", "mean"),
            average_expected_reward=("average_expected_reward", "mean"),
            normalized_pseudo_regret=("normalized_pseudo_regret", "mean"),
            violations=("violations", "mean"),
            updates=("updates", "mean"),
        )
    )

    summary_results = (
        user_results
        .groupby(["T", "policy"], as_index=False)
        .agg(
            mean_offline_value=("offline_value", "mean"),
            std_offline_value=("offline_value", "std"),
            mean_average_offline_reward=("average_offline_reward", "mean"),
            std_average_offline_reward=("average_offline_reward", "std"),
            mean_expected_reward=("average_expected_reward", "mean"),
            std_expected_reward=("average_expected_reward", "std"),
            mean_normalized_regret=("normalized_pseudo_regret", "mean"),
            std_normalized_regret=("normalized_pseudo_regret", "std"),
            mean_violations=("violations", "mean"),
            std_violations=("violations", "std"),
            mean_updates=("updates", "mean"),
            std_updates=("updates", "std"),
        )
    )

    summary_file = kuairand_config.output_dir / "experiment_summary.csv"
    summary_results.to_csv(summary_file, index=False)
    print(summary_results.to_string(index=False))


if __name__ == "__main__":
    run()