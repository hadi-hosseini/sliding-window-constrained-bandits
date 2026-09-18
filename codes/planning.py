import os
import itertools
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

os.makedirs("imgs", exist_ok=True)

COLORS = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "black":  "#333333",
    "gray":   "#8A8A8A",
}


ACTIONS = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
])

ACTION_NAMES = ["0", "e1", "e2"]
D = 2
W = 4

COST_VECTOR = np.array([1.0, 3.0,])
POSITION_WEIGHTS = np.array([4.0, 2.0, 1.0, 1.0,])
BUDGET = 14.0
ACTION_COSTS = (ACTIONS @ COST_VECTOR)
THETA_STAR = np.array([0.6, 1.6,])
TRUE_REWARDS = (ACTIONS @ THETA_STAR)
T_VALUES = np.array([512, 1024, 2048, 4096, 8192,])

NOISE_STD = 0.1
LAMBDA_REG = 1.0
DELTA = 0.05
L_BOUND = np.max(np.linalg.norm(ACTIONS, axis=1))
S_BOUND = np.linalg.norm(THETA_STAR)
REWARD_BOUND = (L_BOUND * S_BOUND)
N_RUNS = 30
BASE_SEED = 2026

POLICY_NAMES = ["Rare-update optimistic planning", "Myopic feasible OFUL", "Stationary OFUL"]

POLICY_COLORS = {
    "Rare-update optimistic planning": COLORS["blue"],
    "Myopic feasible OFUL": COLORS["orange"],
    "Stationary OFUL": COLORS["green"],
}
POLICY_MARKERS = {
    "Rare-update optimistic planning": "o",
    "Myopic feasible OFUL": "s",
    "Stationary OFUL": "^",
}


def build_state_model():
    states = [()]
    for length in range(1, W):
        states.extend(itertools.product(range(len(ACTIONS)), repeat=length))

    states = [tuple(state) for state in states]
    state_to_index = {state: index for index, state in enumerate(states)}
    n_states = len(states)
    n_actions = len(ACTIONS)
    feasible = np.zeros((n_states, n_actions), dtype=bool)
    next_state = -np.ones((n_states, n_actions), dtype=int)

    for state_index, state in enumerate(states):
        for action_index in range(n_actions):
            if len(state) < W - 1:
                is_feasible = True

            else:
                window = (state + (action_index,))
                costs = np.array([ACTION_COSTS[index] for index in window])
                weighted_cost = np.dot(POSITION_WEIGHTS, costs)
                is_feasible = (weighted_cost <= BUDGET + 1e-12)

            feasible[state_index, action_index] = is_feasible
            if is_feasible:
                new_state = (state + (action_index,))
                new_state = (new_state[-(W - 1):])
                next_state[state_index, action_index] = (state_to_index[new_state])

    safe_next_state = (next_state.copy())
    safe_next_state[safe_next_state < 0] = n_states

    return {
        "states": states,
        "state_to_index": state_to_index,
        "feasible": feasible,
        "next_state": next_state,
        "safe_next_state": safe_next_state,
        "n_states": n_states,
        "empty_state": state_to_index[()],
    }


STATE_MODEL = (build_state_model())


def confidence_radius(V, lambda_reg=LAMBDA_REG, noise_scale=NOISE_STD, theta_bound=S_BOUND, delta=DELTA):
    d = V.shape[0]
    sign, logdet_v = (np.linalg.slogdet(V))

    if sign <= 0:
        raise ValueError("V must be positive definite.")

    logdet_initial = (d * np.log(lambda_reg))
    log_term = (0.5 * (logdet_v - logdet_initial) + np.log(1.0 / delta))
    beta = (noise_scale * np.sqrt(max(0.0, 2.0 * log_term)) + np.sqrt(lambda_reg) * theta_bound)
    return float(beta)


def optimistic_rewards(V, b):
    theta_hat = (np.linalg.solve(V, b))
    beta = (confidence_radius(V=V))
    V_inv = np.linalg.inv(V)
    values = np.zeros(len(ACTIONS))

    for action_index, action in enumerate(ACTIONS):
        bonus = (beta * np.sqrt(max(0.0, action @ V_inv @ action)))
        values[action_index] = (theta_hat @ action + bonus)

    values = np.clip(values, -REWARD_BOUND, REWARD_BOUND)
    return values


def bellman_table(stage_rewards, horizon):
    n_states = (STATE_MODEL["n_states"])
    safe_next_state = (STATE_MODEL["safe_next_state"])
    values = np.full((horizon + 1, n_states), -np.inf)
    policy = np.full((horizon + 1, n_states), -1, dtype=np.int8)
    values[0, :] = 0.0

    for n in range(1, horizon + 1):
        previous = np.concatenate([values[n - 1], np.array([-np.inf])])
        q_values = (stage_rewards[None, :] + previous[safe_next_state])
        policy[n] = np.argmax(q_values, axis=1)
        values[n] = np.max(q_values, axis=1)

    return (values, policy)


@lru_cache(maxsize=None)
def offline_optimum(T):
    values, _ = (bellman_table(stage_rewards=TRUE_REWARDS, horizon=int(T)))
    empty_state = (STATE_MODEL["empty_state"])
    return float(values[int(T), empty_state])

@lru_cache(maxsize=None)
def viability_values(T):
    values, _ = (bellman_table(stage_rewards=np.zeros(len(ACTIONS)), horizon=int(T)))
    return values

def count_window_violations(action_indices, tolerance=1e-10):
    if len(action_indices) < W:
        return 0

    violations = 0
    for t in range(W - 1, len(action_indices)):
        window = (action_indices[t - W + 1:t + 1])
        costs = np.array([ACTION_COSTS[index] for index in window])
        weighted_cost = np.dot(POSITION_WEIGHTS, costs)

        if (weighted_cost > BUDGET + tolerance):
            violations += 1

    return int(violations)

def run_rare_update_planning(noise):
    T = len(noise)
    V = (LAMBDA_REG * np.eye(D))
    b = np.zeros(D)
    state = (STATE_MODEL["empty_state"])
    next_state = (STATE_MODEL["next_state"])
    action_indices = []
    total_true_reward = 0.0
    t = 0
    n_updates = 0

    while t < T:
        _, episode_logdet = (np.linalg.slogdet(V))
        optimistic = (optimistic_rewards(V=V, b=b))
        remaining_at_start = (T - t)
        _, policy = (bellman_table(stage_rewards=optimistic, horizon=remaining_at_start))
        n_updates += 1

        while t < T:
            _, current_logdet = (np.linalg.slogdet(V))
            if (current_logdet > episode_logdet + np.log(2.0) + 1e-12):
                break

            remaining = (T - t)
            action_index = int(policy[remaining, state])
            if action_index < 0:
                raise RuntimeError("Planner reached a non-viable state.")

            selected_next_state = (next_state[state, action_index])
            if selected_next_state < 0:
                raise RuntimeError("Planner selected an infeasible action.")

            action = (ACTIONS[action_index])
            action_indices.append(action_index)
            true_reward = (THETA_STAR @ action)
            observation = (true_reward + noise[t])
            total_true_reward += (true_reward)
            V += np.outer(action, action)
            b += (action * observation)
            state = (selected_next_state)
            t += 1

    offline_value = (offline_optimum(T))
    regret = (offline_value - total_true_reward)

    return {
        "regret": float(regret),
        "normalized_regret": float(regret / T),
        "updates": int(n_updates),
    }


def run_myopic_feasible_oful(noise):
    T = len(noise)
    V = (LAMBDA_REG * np.eye(D))
    b = np.zeros(D)
    state = (STATE_MODEL["empty_state"])
    next_state = (STATE_MODEL["next_state"])
    viability = (viability_values(T))
    action_indices = []
    total_true_reward = 0.0
    t = 0
    n_updates = 0

    while t < T:
        _, episode_logdet = (np.linalg.slogdet(V))
        optimistic = (optimistic_rewards(V=V, b=b))
        n_updates += 1

        while t < T:
            _, current_logdet = (np.linalg.slogdet(V))
            if (current_logdet > episode_logdet + np.log(2.0) + 1e-12):
                break

            remaining = (T - t)
            candidates = []
            for action_index in range(len(ACTIONS)):
                candidate_state = (next_state[state, action_index])
                if candidate_state < 0:
                    continue

                if np.isfinite(viability[remaining - 1, candidate_state]):
                    candidates.append(action_index)

            if len(candidates) == 0:
                raise RuntimeError(
                    f"No viable action available "
                    f"at t={t}, "
                    f"state="
                    f"{STATE_MODEL['states'][state]}."
                )

            action_index = max(candidates, key=lambda index: optimistic[index])
            selected_next_state = (next_state[state, action_index])
            if selected_next_state < 0:
                raise RuntimeError(
                    "Myopic policy selected "
                    "an infeasible action."
                )

            action = (ACTIONS[action_index])
            action_indices.append(action_index)
            true_reward = (THETA_STAR @ action)
            observation = (true_reward + noise[t])
            total_true_reward += (true_reward)
            V += np.outer(action, action)
            b += (action * observation)
            state = (selected_next_state)
            t += 1

    offline_value = (offline_optimum(T))
    regret = (offline_value - total_true_reward)

    return {
        "regret": float(regret),
        "normalized_regret": float(regret / T),
        "updates": int(n_updates),
    }


def stationary_action_indices():
    stationary = []
    total_position_weight = np.sum(POSITION_WEIGHTS)
    for action_index in range(len(ACTIONS)):
        cost = (total_position_weight * ACTION_COSTS[action_index])
        if (cost <= BUDGET + 1e-12):
            stationary.append(action_index)

    return stationary

STATIONARY_ACTIONS = (stationary_action_indices())

def run_stationary_oful(noise):
    T = len(noise)
    V = (LAMBDA_REG * np.eye(D))
    b = np.zeros(D)
    action_indices = []
    total_true_reward = 0.0
    t = 0
    n_updates = 0

    while t < T:
        _, episode_logdet = (np.linalg.slogdet(V))
        optimistic = (optimistic_rewards(V=V, b=b))
        action_index = max(STATIONARY_ACTIONS, key=lambda index:optimistic[index])
        action = (ACTIONS[action_index])
        n_updates += 1

        while t < T:
            _, current_logdet = (np.linalg.slogdet(V))

            if (current_logdet > episode_logdet + np.log(2.0) + 1e-12):
                break

            action_indices.append(action_index)
            true_reward = (THETA_STAR @ action)
            observation = (true_reward + noise[t])
            total_true_reward += (true_reward)
            V += np.outer(action, action)
            b += (action * observation)
            t += 1

    offline_value = (offline_optimum(T))
    regret = (offline_value - total_true_reward)

    return {
        "regret": float(regret),
        "normalized_regret": float(regret / T),
        "updates": int(n_updates),
    }


def print_problem_summary():
    print("\nPlanning experiment")
    print("-" * 70)
    print(f"Window length: w = {W}")
    print("Stationary feasible actions: " f"{[ACTION_NAMES[i] for i in STATIONARY_ACTIONS]}")
    best_stationary_reward = max([TRUE_REWARDS[index] for index in STATIONARY_ACTIONS])

    print("Best stationary reward: " f"{best_stationary_reward:.4f}")
    periodic = [1, 2, 0, 1]
    periodic_average_reward = np.mean([TRUE_REWARDS[index] for index in periodic])

    print(
        "Reward of feasible periodic pattern "
        "(e1,e2,0,e1): "
        f"{periodic_average_reward:.4f}"
    )

    periodic_test = (periodic * 10)
    periodic_violations = (count_window_violations(periodic_test))

    print("Periodic-pattern violations: " f"{periodic_violations}")
    print("\nOffline benchmark values:")

    for T in T_VALUES:
        value = (offline_optimum(int(T)))

        print(
            f"T={T:5d}: "
            f"V_off={value:9.4f}, "
            f"average={value/T:.4f}"
        )


def run_planning_experiment(n_runs=N_RUNS, seed=BASE_SEED):
    metrics = ["normalized_regret", "updates"]
    results = {policy: {metric: np.zeros((len(T_VALUES), n_runs)) for metric in metrics} for policy in POLICY_NAMES}
    max_horizon = int(np.max(T_VALUES))

    for run in tqdm(range(n_runs), desc="Runs"):
        rng = (np.random.default_rng(seed + run))
        full_noise = (rng.normal(loc=0.0, scale=NOISE_STD, size=max_horizon))

        for horizon_index, T in enumerate(T_VALUES):
            T = int(T)
            noise = (full_noise[:T])
            planning_result = (run_rare_update_planning(noise=noise))
            myopic_result = (run_myopic_feasible_oful(noise=noise))
            stationary_result = (run_stationary_oful(noise=noise))

            policy_results = {
                "Rare-update optimistic planning": planning_result,
                "Myopic feasible OFUL": myopic_result,
                "Stationary OFUL": stationary_result,
            }

            for policy in POLICY_NAMES:
                for metric in metrics:
                    results[policy][metric][horizon_index, run] = (policy_results[policy][metric])

    return results


def mean_and_ci(values):
    mean = np.mean(values, axis=1)
    if values.shape[1] == 1:
        ci = np.zeros_like(mean)
    else:
        standard_error = (np.std(values, axis=1, ddof=1) / np.sqrt(values.shape[1]))
        ci = (1.96 * standard_error)

    return (mean, ci)


def print_experiment_summary(results):
    for horizon_index, T in enumerate(T_VALUES):
        print(f"\nT = {T}")
        print("-" * 85)

        for policy in POLICY_NAMES:
            regret_data = (results[policy]["normalized_regret"][horizon_index])
            update_data = (results[policy]["updates"][horizon_index])
            regret_mean = np.mean(regret_data)
            update_mean = np.mean(update_data)

            if len(regret_data) > 1:
                regret_ci = (1.96 * np.std(regret_data, ddof=1) / np.sqrt(len(regret_data)))
                update_ci = (1.96 * np.std(update_data, ddof=1) / np.sqrt(len(update_data)))

            else:
                regret_ci = 0.0
                update_ci = 0.0

            print(
                f"{policy:35s} | "
                f"regret/T "
                f"{regret_mean:8.4f} "
                f"± {regret_ci:7.4f} | "
                f"updates "
                f"{update_mean:7.2f} "
                f"± {update_ci:5.2f}"
            )


def plot_planning_experiment(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ax1 = axes[0]
    for policy in POLICY_NAMES:
        mean, ci = (mean_and_ci(results[policy]["normalized_regret"]))

        ax1.plot(T_VALUES, mean, color=POLICY_COLORS[policy], marker=POLICY_MARKERS[policy],
            markersize=7,
            linewidth=2.1,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=policy)

        ax1.fill_between(T_VALUES, mean - ci, mean + ci, color=POLICY_COLORS[policy], alpha=0.30)

    ax1.set_title("(a) Normalized Pseudo-Regret", fontsize=14, pad=12)
    ax1.set_xlabel("Horizon T", fontsize=13)
    ax1.set_ylabel("Pseudo-regret / T", fontsize=13)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(T_VALUES)
    ax1.set_xticklabels([str(T) for T in T_VALUES])
    ax1.tick_params(axis="both", labelsize=10)

    ax1.grid(linestyle="--", linewidth=0.7, alpha=0.22)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_linewidth(0.8)
    ax1.spines["bottom"].set_linewidth(0.8)
    ax1.set_ylim(bottom=0)

    ax2 = axes[1]
    for policy in POLICY_NAMES:
        mean, ci = (mean_and_ci(results[policy]["updates"]))

        ax2.plot(
            T_VALUES,
            mean,
            color=POLICY_COLORS[
                policy
            ],
            marker=POLICY_MARKERS[
                policy
            ],
            markersize=7,
            linewidth=2.1,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=policy
        )

        ax2.fill_between(
            T_VALUES,
            mean - ci,
            mean + ci,
            color=POLICY_COLORS[
                policy
            ],
            alpha=0.30
        )

    ax2.set_title("(b) Policy Updates", fontsize=14, pad=12)
    ax2.set_xlabel("Horizon T", fontsize=13)
    ax2.set_ylabel("Number of policy updates", fontsize=13)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(T_VALUES)
    ax2.set_xticklabels([str(T) for T in T_VALUES])
    ax2.tick_params(axis="both", labelsize=10)

    ax2.grid(linestyle="--", linewidth=0.7, alpha=0.22)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_linewidth(0.8)
    ax2.spines["bottom"].set_linewidth(0.8)
    ax2.set_ylim(bottom=0)
    handles, labels = (ax1.get_legend_handles_labels())

    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.92], w_pad=3.0)
    fig.savefig("imgs/planning.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print_problem_summary()
    results = (run_planning_experiment(n_runs=N_RUNS, seed=BASE_SEED))
    print_experiment_summary(results)
    plot_planning_experiment(results)