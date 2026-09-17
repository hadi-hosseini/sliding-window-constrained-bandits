from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
from tqdm import tqdm

from config import KuaiRandConfig

kuairand_config = KuaiRandConfig()

ABLATION_N_USERS = 50
ABLATION_N_RUNS = 10
ABLATION_T_VALUES = [500, 1000, 2500, 5000]
ABLATION_WINDOW_SIZE = 20
ABLATION_PLANNING_HORIZON = 5

top_categories = pd.read_csv(kuairand_config.data_dir / "top_categories.csv")
preferences = pd.read_csv(kuairand_config.data_dir / "user_category_preferences.csv")
coverage = pd.read_csv(kuairand_config.data_dir / "user_coverage.csv")
K = len(top_categories)
theta_norm_bound = np.sqrt(K)
ABLATION_METHODS = ["Optimistic", "No Optimism",]
FIG_DIR = Path("./imgs")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = (kuairand_config.output_dir / "optimism_ablation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def select_eligible_users(coverage, K, config):
    eligible = coverage[(coverage["num_categories"] == K) & (coverage["min_category_interactions"]>= config.min_category_interactions)]
    user_ids = eligible["user_id"].to_numpy()
    rng = np.random.default_rng(config.random_seed)
    user_ids = rng.choice(user_ids, size=ABLATION_N_USERS, replace=False)
    return user_ids

def get_theta(user_id, preferences, config):
    user_data = (preferences[preferences["user_id"] == user_id].sort_values("action_id"))
    theta_star = (user_data[config.theta_column].astype(float).to_numpy())
    return theta_star


def offline_value(theta_star, T, config):
    w = config.window_size
    c = config.max_exposure
    number_of_categories = w // c
    best_categories = (np.sort(theta_star)[::-1][:number_of_categories])
    value_per_window = (c * best_categories.sum())
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
        return [action for action in range(self.K) if counts[action] < self.c]

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

        return (reward, expected_reward, violation)


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
        return (self.b / self.v_diag)

    def logdet(self):
        return float(np.log(self.v_diag).sum())

    def beta(self):
        logdet_v = self.logdet()
        logdet_lambda = (self.K * np.log(self.lambda_reg))
        log_term = (0.5 * (logdet_v - logdet_lambda) + np.log(1.0 / self.delta))
        beta = (self.noise_r * np.sqrt(2.0 * max(log_term, 0.0)) + np.sqrt(self.lambda_reg) * self.theta_norm_bound)
        return float(beta)

    def optimistic_scores(self):
        theta_hat = self.theta_hat()
        beta = self.beta()
        bonus = (beta / np.sqrt(self.v_diag))
        return (theta_hat + bonus)

    def update(self, action, reward):
        action = int(action)
        self.v_diag[action] += 1.0
        self.b[action] += reward


class RareUpdatePlanning:

    def __init__(self, K, config, theta_norm_bound, use_optimism=True, seed=0):
        self.K = K
        self.w = config.window_size
        self.c = config.max_exposure
        self.det_factor = config.det_factor
        self.planning_horizon = config.planning_horizon
        self.use_optimism = use_optimism
        self.rng = np.random.default_rng(seed)
        self.model = OptimisticLinearBandit(K=K, config=config, theta_norm_bound=theta_norm_bound)
        self.episode_logdet = None
        self.frozen_scores = None
        self.n_updates = 0
        self.initial_order = (self.rng.permutation(K))

    def start_new_episode(self):
        self.episode_logdet = (self.model.logdet())

        if self.use_optimism:
            self.frozen_scores = (self.model.optimistic_scores().copy())
        else:
            self.frozen_scores = (self.model.theta_hat().copy())

        self.n_updates += 1
        self.planned_value.cache_clear()

    def determinant_doubled(self):
        if self.episode_logdet is None:
            return True

        return (self.model.logdet() > self.episode_logdet + np.log(self.det_factor))

    def initialization_action(self, t, feasible_actions):
        index = (t - 1) % self.K
        preferred = int(self.initial_order[index])

        if preferred in feasible_actions:
            return preferred

        return int(self.rng.choice(feasible_actions))

    def transition(self, state, action):
        next_state = (list(state) + [int(action)])
        return tuple(next_state[ -(self.w - 1):])

    def feasible_actions_from_state(self, state):
        counts = Counter(state)
        return [action for action in range(self.K) if counts[action] < self.c]

    @lru_cache(maxsize=None)
    def planned_value(self, state, horizon):
        if horizon <= 0:
            return 0.0

        feasible = (self.feasible_actions_from_state(state))
        if len(feasible) == 0:
            return 0.0

        best_value = -np.inf
        for action in feasible:
            next_state = (self.transition(state, action))
            value = (self.frozen_scores[action] + self.planned_value(next_state, horizon - 1))
            best_value = max(best_value, value)

        return best_value

    def planning_action(self, state, horizon):
        feasible = (self.feasible_actions_from_state(state))

        if len(feasible) == 0:
            return None

        best_value = -np.inf
        candidates = []
        for action in feasible:
            next_state = (self.transition(state, action))
            value = (self.frozen_scores[action])

            if horizon > 1:
                value += (self.planned_value(next_state, horizon - 1))

            if (value > best_value and not np.isclose(value, best_value)):
                best_value = value
                candidates = [action]

            elif np.isclose(value, best_value,):
                candidates.append(action)

        return int(self.rng.choice(candidates))

    def choose(self, state, t, T):
        feasible_actions = (self.feasible_actions_from_state(state))

        if len(feasible_actions) == 0:
            return None

        if t < self.w:
            return (self.initialization_action(t, feasible_actions))

        if self.determinant_doubled():
            self.start_new_episode()

        remaining_steps = T - t + 1
        horizon = min(remaining_steps, self.planning_horizon)
        return self.planning_action(state, horizon)

    def update(self, action, reward):
        if action is not None:
            self.model.update(action, reward)

def run_policy(theta_star, method, T, seed, config):
    env = RollingExposureEnvironment(theta_star=theta_star, config=config, seed=seed)
    use_optimism = (method == "Optimistic")

    agent = RareUpdatePlanning(K=len(theta_star), config=config, theta_norm_bound=theta_norm_bound, use_optimism=use_optimism, seed=seed + 1000)
    for t_round in range(1, T + 1):
        state = tuple(env.history[-(config.window_size - 1):])
        action = agent.choose(state=state, t=t_round, T=T)
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


def mean_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)

    if n == 0:
        return np.nan, np.nan

    mean = float(np.mean(values))

    if n == 1:
        return mean, 0.0

    sem = (np.std(values, ddof=1) / np.sqrt(n))
    critical = t.ppf(1.0 - (1.0 - confidence) / 2.0, df=n - 1)
    return (mean, float(critical * sem))


def run():
    results = []

    selected_user_ids = (select_eligible_users(coverage, K, kuairand_config))

    print(f"Users: {ABLATION_N_USERS}")
    print(f"Runs: {ABLATION_N_RUNS}")
    print(f"T values: {ABLATION_T_VALUES}")
    print(f"Window size W: {ABLATION_WINDOW_SIZE}")
    print(f"Max exposure c: " f"{kuairand_config.max_exposure}")
    print(f"Planning horizon H: " f"{ABLATION_PLANNING_HORIZON}")

    for (user_index, user_id) in enumerate(
        tqdm(selected_user_ids, desc="Optimism ablation", unit="user")):
        theta_star = get_theta(user_id, preferences, kuairand_config)

        for T_value in ABLATION_T_VALUES:
            for run_idx in range(ABLATION_N_RUNS):
                seed = (kuairand_config.random_seed + 100000 * user_index + 100 * run_idx)

                for method in ABLATION_METHODS:
                    outcome = run_policy(theta_star=theta_star, method=method, T=T_value, seed=seed, config=kuairand_config)
                    results.append({
                            "user_id": user_id,
                            "run_idx": run_idx,
                            "T": T_value,
                            "window_size": ABLATION_WINDOW_SIZE,
                            "max_exposure": kuairand_config.max_exposure,
                            "planning_horizon": ABLATION_PLANNING_HORIZON,
                            "method": method,
                            **outcome,
                        })

    results_df = pd.DataFrame(results)
    results_file = (OUTPUT_DIR / "optimism_ablation_results.csv")
    results_df.to_csv(results_file, index=False)
    user_results = (results_df.groupby(["user_id", "T", "method"], as_index=False).agg(
            average_expected_reward=("average_expected_reward", "mean"),
            normalized_pseudo_regret=("normalized_pseudo_regret", "mean"),
            violations=("violations", "mean"),
            updates=("updates", "mean")))

    rows = []

    for T_value in sorted(user_results["T"].unique()):
        for method in ABLATION_METHODS:
            data = user_results[(user_results["T"] == T_value) & (user_results["method"] == method)]
            reward_mean, reward_ci = (mean_ci(data["average_expected_reward"]))
            regret_mean, regret_ci = (mean_ci(data["normalized_pseudo_regret"]))
            violation_mean, violation_ci = (mean_ci(data["violations"]))
            updates_mean, updates_ci = (mean_ci(data["updates"]))
            rows.append({
                    "T": T_value,
                    "method": method,
                    "mean_reward": reward_mean,
                    "reward_ci95": reward_ci,
                    "mean_regret": regret_mean,
                    "regret_ci95": regret_ci,
                    "mean_violations": violation_mean,
                    "violations_ci95": violation_ci,
                    "mean_updates": updates_mean,
                    "updates_ci95": updates_ci,
                })

    summary = pd.DataFrame(rows)
    summary_file = (OUTPUT_DIR / "optimism_ablation_summary.csv")
    summary.to_csv(summary_file, index=False)
    print(summary.to_string(index=False))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for method in ABLATION_METHODS:
        data = (summary[summary["method"] == method].sort_values("T"))
        axes[0].errorbar(data["T"], data["mean_reward"], yerr=data["reward_ci95"], marker="o", capsize=3, label=method,)
        axes[1].errorbar(data["T"], data["mean_regret"], yerr=data["regret_ci95"], marker="o", capsize=3, label=method)

    axes[0].set_xlabel("Horizon $T$")
    axes[0].set_ylabel("Average Reward")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Horizon $T$")
    axes[1].set_ylabel("Normalized Pseudo-Regret")
    axes[1].grid(True, alpha=0.3)
    handles, labels = (axes[0].get_legend_handles_labels())
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    figure_file = (FIG_DIR / "optimism_ablation.png")
    plt.savefig(figure_file, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", results_file)
    print("Saved:", summary_file)
    print("Saved:", figure_file)

if __name__ == "__main__":
    run()