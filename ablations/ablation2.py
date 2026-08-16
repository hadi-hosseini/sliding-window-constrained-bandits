import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from rare_switching import (
    THETA_STAR,
    W,
    T,
    RHO,
    NOISE_STD,
    LAMBDA_REG,
    DELTA,
    optimistic_simplex_action,
    transition_path,
    count_window_violations,
)

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

ALPHA = 2.0
N_RUNS = 30
SEED = 42

TRANSITION_MODES = [
    "proposed",
    "center-detour",
]

MODE_LABELS = {
    "proposed": "Proposed",
    "center-detour": "Center detour",
}

MODE_COLORS = {
    "proposed": COLORS["blue"],
    "center-detour": COLORS["orange"],
}


def safe_segment(z, z_next, w, rho):
    distance = np.linalg.norm(z_next - z, ord=1)

    if distance <= 1e-12:
        return []

    n_steps = int(np.ceil((w - 1) * distance / rho))
    n_steps = max(n_steps, 1)

    path = []

    for j in range(1, n_steps + 1):
        fraction = j / n_steps
        action = (1.0 - fraction) * z + fraction * z_next
        path.append(action.copy())

    return path


def alternative_transition_path(z, z_next, w, rho):
    if np.allclose(z, z_next, atol=1e-12, rtol=0.0):
        return []

    d = len(z)
    center = np.full(d, 1.0 / d)

    path = []

    path.extend(
        safe_segment(
            z=z,
            z_next=center,
            w=w,
            rho=rho,
        )
    )

    path.extend(
        safe_segment(
            z=center,
            z_next=z_next,
            w=w,
            rho=rho,
        )
    )

    for _ in range(w - 2):
        path.append(z_next.copy())

    return path


def run_transition_ablation(noise, transition_mode):
    T_current = len(noise)
    d = len(THETA_STAR)

    V = LAMBDA_REG * np.eye(d)
    b = np.zeros(d)
    actions = np.zeros((T_current, d))

    optimal_reward = np.max(THETA_STAR)

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
            theta_bound=np.linalg.norm(THETA_STAR),
            delta=DELTA,
        )

        if (
            previous_target is not None
            and not np.allclose(
                z,
                previous_target,
                atol=1e-12,
                rtol=0.0,
            )
        ):
            target_switches += 1

            if transition_mode == "proposed":
                path = transition_path(
                    z=previous_target,
                    z_next=z,
                    w=W,
                    rho=RHO,
                )

            elif transition_mode == "center-detour":
                path = alternative_transition_path(
                    z=previous_target,
                    z_next=z,
                    w=W,
                    rho=RHO,
                )

            else:
                raise ValueError(
                    f"Unknown transition mode: {transition_mode}"
                )

            for action in path:
                if t >= T_current:
                    break

                actions[t] = action

                pseudo_regret += (
                    optimal_reward
                    - np.dot(THETA_STAR, action)
                )

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
                current_logdet
                <= episode_logdet + np.log(ALPHA) + 1e-12
            )

            warmup_condition = (
                episode == 1
                and stationary_rounds < W - 1
            )

            if not (determinant_condition or warmup_condition):
                break

            action = z
            actions[t] = action

            reward = (
                np.dot(THETA_STAR, action)
                + noise[t]
            )

            pseudo_regret += (
                optimal_reward
                - np.dot(THETA_STAR, action)
            )

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
    metrics = [
        "regret",
        "switches",
        "transition_rounds",
        "violations",
    ]

    results = {
        mode: {
            metric: np.zeros(N_RUNS)
            for metric in metrics
        }
        for mode in TRANSITION_MODES
    }

    for run in tqdm(
        range(N_RUNS),
        desc="Transition-mechanism ablation",
    ):
        rng = np.random.default_rng(SEED + run)

        noise = rng.normal(
            0.0,
            NOISE_STD,
            T,
        )

        for mode in TRANSITION_MODES:
            result = run_transition_ablation(
                noise=noise,
                transition_mode=mode,
            )

            for metric in metrics:
                results[mode][metric][run] = result[metric]

    return results


def print_results(results):
    print("\nTransition-mechanism ablation\n")

    for mode in TRANSITION_MODES:
        regret, regret_ci = mean_and_ci(
            results[mode]["regret"]
        )

        switches, switches_ci = mean_and_ci(
            results[mode]["switches"]
        )

        transitions, transitions_ci = mean_and_ci(
            results[mode]["transition_rounds"]
        )

        violations, violations_ci = mean_and_ci(
            results[mode]["violations"]
        )

        print(
            f"{MODE_LABELS[mode]:>14} | "
            f"regret={regret:8.2f} +/- {regret_ci:6.2f} | "
            f"switches={switches:7.2f} +/- {switches_ci:5.2f} | "
            f"transitions={transitions:8.2f} +/- {transitions_ci:6.2f} | "
            f"violations={violations:6.2f} +/- {violations_ci:5.2f}"
        )


def collect_plot_data(results, metric):
    means = []
    cis = []

    for mode in TRANSITION_MODES:
        mean, ci = mean_and_ci(
            results[mode][metric]
        )

        means.append(mean)
        cis.append(ci)

    return np.asarray(means), np.asarray(cis)


def format_axis(ax):
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.22,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        labelsize=10,
    )

    ax.set_axisbelow(True)


def plot_metric(
    ax,
    x,
    means,
    cis,
    title,
    ylabel,
):
    colors = [
        MODE_COLORS[mode]
        for mode in TRANSITION_MODES
    ]

    ax.bar(
        x,
        means,
        width=0.55,
        color=colors,
        edgecolor="white",
        linewidth=0.9,
        alpha=0.95,
        zorder=3,
    )

    ax.errorbar(
        x,
        means,
        yerr=cis,
        fmt="none",
        ecolor=COLORS["black"],
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        zorder=4,
    )

    ax.set_title(
        title,
        fontsize=13,
        pad=9,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=12,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            MODE_LABELS[mode]
            for mode in TRANSITION_MODES
        ],
        fontsize=11,
    )

    upper = np.max(means + cis)

    ax.set_ylim(
        0,
        upper * 1.14,
    )

    format_axis(ax)


def plot_results(results):
    x = np.arange(len(TRANSITION_MODES))

    regret_means, regret_cis = collect_plot_data(
        results,
        "regret",
    )

    transition_means, transition_cis = collect_plot_data(
        results,
        "transition_rounds",
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.2, 3.7),
    )

    plot_metric(
        ax=axes[0],
        x=x,
        means=regret_means,
        cis=regret_cis,
        title="(a) Pseudo-Regret",
        ylabel="Pseudo-regret",
    )

    plot_metric(
        ax=axes[1],
        x=x,
        means=transition_means,
        cis=transition_cis,
        title="(b) Transition Rounds",
        ylabel="Transition rounds",
    )

    fig.tight_layout(
        w_pad=2.8
    )

    fig.savefig(
        "imgs/ablation2.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    results = run_ablation()
    print_results(results)
    plot_results(results)