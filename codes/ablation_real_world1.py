from pathlib import Path
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import KuaiRandConfig


kuairand_config = KuaiRandConfig()

top_categories = pd.read_csv(kuairand_config.data_dir / "top_categories.csv")
preferences = pd.read_csv(kuairand_config.data_dir / "user_category_preferences.csv")
coverage = pd.read_csv(kuairand_config.data_dir / "user_coverage.csv")
K = len(top_categories)
theta_norm_bound = np.sqrt(K)

PLANNING_HORIZONS = [1, 2, 5, 10, "full"]
ABLATION_T = 200
WINDOW_SIZE = 4
ABLATION_N_USERS = 100
ABLATION_N_RUNS = 5
MAX_EXPOSURE = 1


ABLATION_OUTPUT_DIR = Path("./imgs")
ABLATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def select_eligible_users(coverage, K, config, n_users):
    eligible = coverage[(coverage["num_categories"] == K) & (coverage["min_category_interactions"] >= config.min_category_interactions)]
    user_ids = eligible["user_id"].to_numpy()
    rng = np.random.default_rng(config.random_seed)
    user_ids = rng.choice(user_ids, size=n_users, replace=False)
    return user_ids

def get_theta(user_id, preferences, config):
    user_data = preferences[preferences["user_id"] == user_id].sort_values("action_id")
    theta_star = user_data[config.theta_column].astype(float).to_numpy()
    return theta_star


def offline_value(theta_star, T):
    c = MAX_EXPOSURE
    number_of_categories = WINDOW_SIZE // c
    best_categories = np.sort(theta_star)[::-1][:number_of_categories]
    value_per_window = c * best_categories.sum()
    return (T // WINDOW_SIZE) * value_per_window


class RollingExposureEnvironment:
    def __init__(self, theta_star, seed=0):
        self.theta = np.asarray(theta_star, dtype=float)
        self.K = len(self.theta)
        self.w = WINDOW_SIZE
        self.c = MAX_EXPOSURE
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
        log_term = (0.5 * (logdet_v - logdet_lambda) + np.log(1.0 / self.delta))
        beta = (self.noise_r * np.sqrt(2.0 * max(log_term, 0.0)) + np.sqrt(self.lambda_reg) * self.theta_norm_bound)
        return float(beta)

    def optimistic_scores(self):
        theta_hat = self.theta_hat()
        beta = self.beta()
        bonus = beta / np.sqrt(self.v_diag)
        return theta_hat + bonus

    def update(self, action, reward):
        action = int(action)
        self.v_diag[action] += 1.0
        self.b[action] += reward



class RareUpdateOptimisticPlanning:

    def __init__(self, K, config, theta_norm_bound, planning_horizon, seed=0):
        self.K = K
        self.w = WINDOW_SIZE
        self.c = MAX_EXPOSURE
        self.det_factor = config.det_factor
        self.rng = np.random.default_rng(seed)
        self.model = OptimisticLinearBandit(K=K, config=config, theta_norm_bound=theta_norm_bound)
        self.planning_horizon = planning_horizon
        self.episode_logdet = None
        self.frozen_scores = None
        self.n_updates = 0
        self.initial_order = self.rng.permutation(K)

    def start_new_episode(self):
        self.episode_logdet = self.model.logdet()
        self.frozen_scores = self.model.optimistic_scores()
        self.n_updates += 1
        self.optimistic_value.cache_clear()

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

    def normalize_state(self, state):
        return tuple(state[-(self.w - 1):])

    def transition(self, state, action):
        state = list(state)
        state.append(int(action))
        return tuple(state[-(self.w - 1):])

    def feasible_actions_from_state(self, state):
        counts = Counter(state)
        return [action for action in range(self.K) if counts[action] < self.c]

    @lru_cache(maxsize=None)
    def optimistic_value(self, state, steps_remaining):
        if steps_remaining <= 0:
            return 0.0

        feasible = self.feasible_actions_from_state(state)

        if len(feasible) == 0:
            return -np.inf

        best_value = -np.inf
        for action in feasible:
            next_state = self.transition(state, action)
            value = (self.frozen_scores[action] + self.optimistic_value(next_state, steps_remaining - 1))

            if value > best_value:
                best_value = value

        return float(best_value)

    def planning_action(self, state, planning_steps):
        feasible = self.feasible_actions_from_state(state)

        if len(feasible) == 0:
            return None

        best_value = -np.inf
        candidates = []

        for action in feasible:
            next_state = self.transition(state, action)
            value = self.frozen_scores[action]

            if planning_steps > 1:
                continuation = self.optimistic_value(next_state, planning_steps - 1)
                value += continuation

            if value > best_value and not np.isclose(value, best_value):
                best_value = value
                candidates = [action]

            elif np.isclose(value, best_value):
                candidates.append(action)

        return int(self.rng.choice(candidates))

    def choose(self, state, t, T):
        state = self.normalize_state(state)
        feasible_actions = self.feasible_actions_from_state(state)

        if len(feasible_actions) == 0:
            return None

        if t < self.w:
            return self.initialization_action(t=t, feasible_actions=feasible_actions)

        if self.determinant_doubled():
            self.start_new_episode()

        remaining_steps = T - t + 1
        if self.planning_horizon == "full":
            planning_steps = remaining_steps
        else:
            H = int(self.planning_horizon)
            planning_steps = min(H, remaining_steps)

        return self.planning_action(state=state, planning_steps=planning_steps)

    def update(self, action, reward):
        if action is not None:
            self.model.update(action, reward)



def run_planning_horizon_policy(theta_star, planning_horizon, T, seed, config):
    env = RollingExposureEnvironment(theta_star=theta_star, seed=seed)
    agent = RareUpdateOptimisticPlanning(K=len(theta_star), config=config, theta_norm_bound=theta_norm_bound, planning_horizon=planning_horizon, seed=seed + 1000)

    for t in range(1, T + 1):
        state = tuple(env.history[-(WINDOW_SIZE - 1):])
        action = agent.choose(state=state, t=t, T=T)
        reward, _, _ = env.step(action)
        agent.update(action, reward)

    V_off = offline_value(theta_star=theta_star, T=T)
    pseudo_regret = V_off - env.expected_reward

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


def horizon_label(H):
    if H == "full":
        return "Full"

    return f"H={int(H)}"


def run_ablation():
    if ABLATION_T % WINDOW_SIZE != 0:
        raise ValueError("ABLATION_T must be a multiple of window_size.")

    selected_user_ids = select_eligible_users(coverage=coverage, K=K, config=kuairand_config, n_users=ABLATION_N_USERS)
    rows = []

    for user_index, user_id in enumerate(tqdm(selected_user_ids, desc="Planning-horizon ablation", unit="user")):
        theta_star = get_theta(user_id=user_id, preferences=preferences, config=kuairand_config)
        for run_idx in range(ABLATION_N_RUNS):   
            seed = kuairand_config.random_seed + 100000 * user_index + 100 * run_idx

            for H in PLANNING_HORIZONS:
                outcome = run_planning_horizon_policy(theta_star=theta_star, planning_horizon=H, T=ABLATION_T, seed=seed, config=kuairand_config)
                rows.append({
                        "user_id": user_id,
                        "run_idx": run_idx,
                        "T": ABLATION_T,
                        "planning_horizon": horizon_label(H),
                        "planning_horizon_raw": H,
                        **outcome,
                    })

    results_df = pd.DataFrame(rows)

    # results_file = (ABLATION_OUTPUT_DIR/ "planning_horizon_ablation_results.csv")
    # results_df.to_csv(results_file, index=False)

    user_results = (
        results_df
        .groupby(["user_id", "T", "planning_horizon"], as_index=False)
        .agg(average_expected_reward=("average_expected_reward", "mean"),
             normalized_pseudo_regret=("normalized_pseudo_regret", "mean"),
            violations=("violations", "mean"),
            updates=("updates", "mean"),
        ))

    summary = (
        user_results
        .groupby(
            ["T", "planning_horizon"],
            as_index=False,
        )
        .agg(
            mean_reward=(
                "average_expected_reward",
                "mean",
            ),
            std_reward=(
                "average_expected_reward",
                "std",
            ),
            mean_regret=(
                "normalized_pseudo_regret",
                "mean",
            ),
            std_regret=(
                "normalized_pseudo_regret",
                "std",
            ),
            mean_violations=(
                "violations",
                "mean",
            ),
            std_violations=(
                "violations",
                "std",
            ),
            mean_updates=(
                "updates",
                "mean",
            ),
            std_updates=(
                "updates",
                "std",
            ),
            n_users=("user_id", "nunique"),
        )
    )

    summary["reward_ci95"] = (1.96 * summary["std_reward"] / np.sqrt(summary["n_users"]))
    summary["regret_ci95"] = (1.96 * summary["std_regret"] / np.sqrt(summary["n_users"]))
    summary["updates_ci95"] = (1.96 * summary["std_updates"] / np.sqrt(summary["n_users"]))

    order = [horizon_label(H) for H in PLANNING_HORIZONS]
    order_map = {label: i for i, label in enumerate(order)}

    summary["sort_index"] = summary["planning_horizon"].map(order_map)
    summary = (summary.sort_values("sort_index").drop(columns="sort_index").reset_index(drop=True))
    # summary_file = (ABLATION_OUTPUT_DIR / "planning_horizon_ablation_summary.csv")
    # summary.to_csv(summary_file, index=False)

    print("\nPlanning-horizon ablation summary")
    print(summary.to_string(index=False))

    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].errorbar(x, summary["mean_regret"], yerr=summary["regret_ci95"], marker="o", capsize=4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary["planning_horizon"])
    axes[0].set_xlabel("Planning Horizon")
    axes[0].set_ylabel("Normalized Pseudo-Regret")
    axes[0].set_title("Normalized Pseudo-Regret")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].errorbar(x, summary["mean_updates"], yerr=summary["updates_ci95"], marker="o", capsize=4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary["planning_horizon"])
    axes[1].set_xlabel("Planning Horizon")
    axes[1].set_ylabel("Policy Updates")
    axes[1].set_title("Policy Updates")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(f"Planning-Horizon Ablation (T={ABLATION_T})")

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    figure_file = (ABLATION_OUTPUT_DIR / "planning_horizon_ablation.png")
    plt.savefig(figure_file, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", figure_file)

if __name__ == "__main__":
    run_ablation()
