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
    run_rare_switching_oful,
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

ALPHA_VALUES = [1.25, 1.5, 2.0, 4.0, 8.0]
N_RUNS = 30
SEED = 42


def mean_and_ci(values):
    values = np.asarray(values, dtype=float)
    mean = np.mean(values)

    if len(values) <= 1:
        return mean, 0.0

    se = np.std(values, ddof=1) / np.sqrt(len(values))
    return mean, 1.96 * se


def run_switching_threshold_ablation():
    metrics = [
        "regret",
        "switches",
        "transition_rounds",
        "violations",
    ]

    results = {
        alpha: {
            metric: np.zeros(N_RUNS)
            for metric in metrics
        }
        for alpha in ALPHA_VALUES
    }

    for run in tqdm(
        range(N_RUNS),
        desc="Switching-threshold ablation",
    ):
        rng = np.random.default_rng(SEED + run)
        noise = rng.normal(0.0, NOISE_STD, T)

        for alpha in ALPHA_VALUES:
            result = run_rare_switching_oful(
                noise=noise,
                rho=RHO,
                alpha=alpha,
                theta_star=THETA_STAR,
                w=W,
            )

            for metric in metrics:
                results[alpha][metric][run] = result[metric]

    return results


def print_results(results):
    print("\nSwitching-threshold ablation\n")

    for alpha in ALPHA_VALUES:
        regret, regret_ci = mean_and_ci(
            results[alpha]["regret"]
        )

        switches, switches_ci = mean_and_ci(
            results[alpha]["switches"]
        )

        transitions, transitions_ci = mean_and_ci(
            results[alpha]["transition_rounds"]
        )

        violations, violations_ci = mean_and_ci(
            results[alpha]["violations"]
        )

        print(
            f"alpha={alpha:>4.2f} | "
            f"regret={regret:8.2f} +/- {regret_ci:6.2f} | "
            f"switches={switches:7.2f} +/- {switches_ci:5.2f} | "
            f"transitions={transitions:8.2f} +/- {transitions_ci:6.2f} | "
            f"violations={violations:6.2f} +/- {violations_ci:5.2f}"
        )


def collect_plot_data(results, metric):
    means = []
    cis = []

    for alpha in ALPHA_VALUES:
        mean, ci = mean_and_ci(
            results[alpha][metric]
        )
        means.append(mean)
        cis.append(ci)

    return (
        np.asarray(means),
        np.asarray(cis),
    )


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
    color,
    title,
    ylabel,
):
    ax.fill_between(
        x,
        means - cis,
        means + cis,
        color=color,
        alpha=0.15,
        linewidth=0,
    )

    ax.plot(
        x,
        means,
        color=color,
        marker="o",
        markersize=7,
        linewidth=2.2,
        markeredgecolor="white",
        markeredgewidth=0.8,
    )

    ax.set_title(
        title,
        fontsize=12,
        pad=9,
    )

    ax.set_xlabel(
        r"Switching threshold $\alpha$",
        fontsize=11,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=11,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["1.25", "1.5", "2", "4", "8"]
    )

    ax.set_xlim(
        -0.35,
        len(ALPHA_VALUES) - 0.65,
    )

    y_min = np.min(means - cis)
    y_max = np.max(means + cis)
    y_range = max(y_max - y_min, 1.0)

    ax.set_ylim(
        max(0.0, y_min - 0.08 * y_range),
        y_max + 0.10 * y_range,
    )

    format_axis(ax)


def plot_results(results):
    x = np.arange(len(ALPHA_VALUES))

    regret_means, regret_cis = collect_plot_data(
        results,
        "regret",
    )

    switch_means, switch_cis = collect_plot_data(
        results,
        "switches",
    )

    transition_means, transition_cis = collect_plot_data(
        results,
        "transition_rounds",
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.5, 3.7),
    )

    plot_metric(
        ax=axes[0],
        x=x,
        means=regret_means,
        cis=regret_cis,
        color=COLORS["blue"],
        title="(a) Pseudo-Regret",
        ylabel="Pseudo-regret",
    )

    plot_metric(
        ax=axes[1],
        x=x,
        means=switch_means,
        cis=switch_cis,
        color=COLORS["orange"],
        title="(b) Target Switches",
        ylabel="Number of target switches",
    )

    plot_metric(
        ax=axes[2],
        x=x,
        means=transition_means,
        cis=transition_cis,
        color=COLORS["green"],
        title="(c) Transition Rounds",
        ylabel="Transition rounds",
    )

    fig.tight_layout(
        w_pad=2.5
    )

    fig.savefig(
        "imgs/ablation1.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    results = run_switching_threshold_ablation()
    print_results(results)
    plot_results(results)