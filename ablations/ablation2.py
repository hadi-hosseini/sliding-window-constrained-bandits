import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from rare_switching import (W, T, RHO, NOISE_STD, LAMBDA_REG, DELTA, optimistic_simplex_action, transition_path, count_window_violations)

os.makedirs("imgs", exist_ok=True)

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#333333",
    "gray": "#8A8A8A",
}

D_VALUES = [4, 8, 16, 32]
ALPHA = 2.0
N_RUNS = 30
SEED = 42
SUBOPTIMAL_REWARD = 0.8
OPTIMAL_REWARD = 1.0

def theta_for_dimension(d):
    theta = np.full(d, SUBOPTIMAL_REWARD, dtype=float)
    theta[-1] = OPTIMAL_REWARD
    return theta

def run_dimension_instance(noise, d):
    theta_star = theta_for_dimension(d)
    theta_bound = np.linalg.norm(theta_star)
    T_current = len(noise)

    V = LAMBDA_REG * np.eye(d)
    b = np.zeros(d)
    actions = np.zeros((T_current, d))

    pseudo_regret = 0.0
    target_switches = 0
    transition_rounds = 0
    t = 0
    episode = 1
    previous_target = None

    while t < T_current:
        sign, episode_logdet = np.linalg.slogdet(V)
        if sign <= 0:
            raise ValueError("V must be positive definite")

        z, _ = optimistic_simplex_action(
            V=V,
            b=b,
            lambda_reg=LAMBDA_REG,
            noise_scale=NOISE_STD,
            theta_bound=theta_bound,
            delta=DELTA,
        )

        if previous_target is not None and not np.allclose(
            z, previous_target, atol=1e-12, rtol=0.0
        ):
            target_switches += 1
            path = transition_path(
                z=previous_target,
                z_next=z,
                w=W,
                rho=RHO,
            )

            for action in path:
                if t >= T_current:
                    break
                actions[t] = action
                pseudo_regret += OPTIMAL_REWARD - np.dot(theta_star, action)
                transition_rounds += 1
                t += 1

        if t >= T_current:
            break

        stationary_rounds = 0

        while t < T_current:
            sign, current_logdet = np.linalg.slogdet(V)
            if sign <= 0:
                raise ValueError("V must be positive definite")

            determinant_condition = (
                current_logdet <= episode_logdet + np.log(ALPHA) + 1e-12
            )
            warmup_condition = (
                episode == 1 and stationary_rounds < W - 1
            )

            if not (determinant_condition or warmup_condition):
                break

            action = z
            actions[t] = action
            reward = np.dot(theta_star, action) + noise[t]

            pseudo_regret += OPTIMAL_REWARD - np.dot(theta_star, action)
            V += np.outer(action, action)
            b += reward * action

            stationary_rounds += 1
            t += 1

        previous_target = z.copy()
        episode += 1

    violations = count_window_violations(
        actions=actions,
        w=W,
        rho=RHO,
    )

    return {
        "regret": float(pseudo_regret),
        "switches": int(target_switches),
        "transition_rounds": int(transition_rounds),
        "violations": int(violations),
    }

def mean_and_ci(values):
    values = np.asarray(values, dtype=float)
    mean = np.mean(values)
    if len(values) <= 1:
        return mean, 0.0
    se = np.std(values, ddof=1) / np.sqrt(len(values))
    return mean, 1.96 * se

def run_ablation():
    metrics = ["regret", "switches", "transition_rounds", "violations"]
    results = {d: {metric: np.zeros(N_RUNS) for metric in metrics} for d in D_VALUES}

    for run in tqdm(range(N_RUNS), desc="Action-dimension ablation"):
        rng = np.random.default_rng(SEED + run)
        noise = rng.normal(0.0, NOISE_STD, T)

        for d in D_VALUES:
            result = run_dimension_instance(noise=noise, d=d)
            for metric in metrics:
                results[d][metric][run] = result[metric]

    return results

def print_results(results):
    print("\nAction-dimension ablation")
    print(f"Fixed reward gap = {OPTIMAL_REWARD - SUBOPTIMAL_REWARD:.2f}\n")

    for d in D_VALUES:
        regret, regret_ci = mean_and_ci(results[d]["regret"])
        switches, switches_ci = mean_and_ci(results[d]["switches"])
        transitions, transitions_ci = mean_and_ci(
            results[d]["transition_rounds"]
        )
        violations, violations_ci = mean_and_ci(results[d]["violations"])

        print(
            f"d={d:2d} | "
            f"regret={regret:8.2f} +/- {regret_ci:6.2f} | "
            f"switches={switches:7.2f} +/- {switches_ci:5.2f} | "
            f"transitions={transitions:8.2f} +/- {transitions_ci:6.2f} | "
            f"violations={violations:6.2f} +/- {violations_ci:5.2f}"
        )

def collect_plot_data(results, metric):
    means, cis = [], []
    for d in D_VALUES:
        mean, ci = mean_and_ci(results[d][metric])
        means.append(mean)
        cis.append(ci)
    return np.asarray(means), np.asarray(cis)

def format_axis(ax):
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_axisbelow(True)

def plot_metric(ax, x, means, cis, title, ylabel, color):
    ax.errorbar(
        x,
        means,
        yerr=cis,
        marker="o",
        markersize=6,
        linewidth=2,
        capsize=4,
        color=color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=3,
    )
    ax.set_title(title, fontsize=13, pad=9)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Action dimension $d$", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in D_VALUES])
    format_axis(ax)

def plot_results(results):
    x = np.arange(len(D_VALUES))

    regret_means, regret_cis = collect_plot_data(results, "regret")
    switch_means, switch_cis = collect_plot_data(results, "switches")
    transition_means, transition_cis = collect_plot_data(
        results, "transition_rounds"
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.6))

    plot_metric(
        axes[0], x, regret_means, regret_cis,
        "(a) Pseudo-Regret", "Pseudo-regret", COLORS["blue"]
    )
    plot_metric(
        axes[1], x, switch_means, switch_cis,
        "(b) Target Switches", "Target switches", COLORS["orange"]
    )
    plot_metric(
        axes[2], x, transition_means, transition_cis,
        "(c) Transition Rounds", "Transition rounds", COLORS["green"]
    )

    fig.tight_layout(w_pad=2.2)
    fig.savefig("imgs/ablation2.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    results = run_ablation()
    print_results(results)
    plot_results(results)