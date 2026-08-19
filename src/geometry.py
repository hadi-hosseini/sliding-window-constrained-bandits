import os
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


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


def solve_offline_lp(T, w, solver="CLARABEL"):

    x = cp.Variable(T)

    constraints = [
        x >= 0,
        x <= 1
    ]

    for t in range(w - 1, T):
        constraints.append(
            cp.sum(x[t - w + 1:t + 1]) <= w / 2
        )

    objective = cp.Maximize(
        cp.sum(x)
    )

    problem = cp.Problem(
        objective,
        constraints
    )

    try:
        problem.solve(
            solver=solver
        )
    except cp.error.SolverError:
        problem.solve()

    if problem.status not in [
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE
    ]:
        raise RuntimeError(
            f"Optimization failed for T={T}, w={w}. "
            f"Status: {problem.status}"
        )

    return (
        float(problem.value),
        np.asarray(x.value).flatten()
    )


def stationary_value(T):

    z_star = 0.5

    return T * z_star


def run_geometry_experiment():

    w_values = [
        4, 6, 8, 10, 12, 14, 16, 18,
        20, 22, 24, 26, 28, 30, 32
    ]

    results = {}

    for w in w_values:

        remainders = []
        horizons = []
        offline_values = []
        stationary_values = []
        gaps = []

        print(f"\nRunning w = {w}")
        print("-" * 65)

        for r in range(w):

            T = 20 * w + r

            V_T, x_opt = solve_offline_lp(
                T=T,
                w=w
            )

            V_stationary = stationary_value(
                T
            )

            delta_T = (
                V_T - V_stationary
            )

            if abs(delta_T) < 1e-7:
                delta_T = 0.0

            remainders.append(r)
            horizons.append(T)
            offline_values.append(V_T)
            stationary_values.append(
                V_stationary
            )
            gaps.append(delta_T)

            print(
                f"r={r:2d}, "
                f"T={T:4d}, "
                f"V_T={V_T:9.4f}, "
                f"stationary={V_stationary:9.4f}, "
                f"gap={delta_T:7.4f}"
            )

        results[w] = {
            "r": np.array(remainders),
            "T": np.array(horizons),
            "V_T": np.array(offline_values),
            "stationary": np.array(
                stationary_values
            ),
            "gap": np.array(gaps),
        }

    return results


def plot_geometry_experiment(
    results,
    w=32
):

    if w not in results:
        raise ValueError(
            f"w={w} is not available. "
            f"Available values: "
            f"{list(results.keys())}"
        )

    data = results[w]

    w_values = np.array(
        sorted(results.keys())
    )

    max_gaps = np.array([
        np.max(
            results[current_w]["gap"]
        )
        for current_w in w_values
    ])

    theoretical_gaps = (
        w_values / 4.0
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.2)
    )

    ax1 = axes[0]

    ax1.plot(
        data["r"],
        data["gap"],
        color=COLORS["blue"],
        marker="o",
        markersize=5.5,
        linewidth=2.4,
        markeredgecolor="white",
        markeredgewidth=0.6,
        label=f"Window length w = {w}",
        zorder=3
    )

    ax1.set_xlabel(
        "Horizon remainder r",
        fontsize=13
    )

    ax1.set_ylabel(
        "Stationary approximation gap Δ",
        fontsize=13
    )

    ax1.set_title(
        "(a) Horizon Misalignment",
        fontsize=14,
        pad=12
    )

    ax1.grid(
        linestyle="--",
        linewidth=0.7,
        alpha=0.22
    )

    ax1.legend(
        frameon=False,
        fontsize=10,
        loc="upper left"
    )

    ax1.tick_params(
        axis="both",
        labelsize=11
    )

    ax1.spines[
        "top"
    ].set_visible(False)

    ax1.spines[
        "right"
    ].set_visible(False)

    ax1.spines[
        "left"
    ].set_linewidth(0.8)

    ax1.spines[
        "bottom"
    ].set_linewidth(0.8)

    ax1.set_ylim(
        bottom=0
    )

    ax1.set_xlim(
        data["r"].min() - 0.5,
        data["r"].max() + 0.5
    )

    ax2 = axes[1]

    ax2.plot(
        w_values,
        theoretical_gaps,
        color=COLORS["black"],
        linestyle="--",
        linewidth=2.0,
        alpha=0.85,
        label="Theoretical tight bound",
        zorder=1
    )

    ax2.plot(
        w_values,
        max_gaps,
        linestyle="None",
        marker="o",
        markersize=8,
        color=COLORS["blue"],
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="Observed maximum gap",
        zorder=3
    )

    ax2.set_xlabel(
        "Window length w",
        fontsize=13
    )

    ax2.set_ylabel(
        "Maximum stationary approximation gap",
        fontsize=13
    )

    ax2.set_title(
        "(b) Tight Dependence on Window Length",
        fontsize=14,
        pad=12
    )

    ax2.set_xticks(
        w_values
    )

    ax2.tick_params(
        axis="x",
        rotation=45,
        labelsize=10
    )

    ax2.tick_params(
        axis="y",
        labelsize=11
    )

    ax2.grid(
        linestyle="--",
        linewidth=0.7,
        alpha=0.22
    )

    ax2.legend(
        frameon=False,
        fontsize=10,
        loc="upper left"
    )

    ax2.spines[
        "top"
    ].set_visible(False)

    ax2.spines[
        "right"
    ].set_visible(False)

    ax2.spines[
        "left"
    ].set_linewidth(0.8)

    ax2.spines[
        "bottom"
    ].set_linewidth(0.8)

    ax2.set_ylim(
        bottom=0
    )

    fig.suptitle(
        "Offline Geometry and Stationary Approximation Gap",
        fontsize=15,
        y=1.02
    )

    fig.tight_layout(
        w_pad=3.0
    )

    fig.savefig(
        "imgs/experiment1_geometry_combined.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        "\nObserved vs. theoretical maximum gaps"
    )

    print("-" * 60)

    for current_w, observed, theory in zip(
        w_values,
        max_gaps,
        theoretical_gaps
    ):

        print(
            f"w={current_w:2d}, "
            f"observed={observed:6.4f}, "
            f"theoretical={theory:6.4f}, "
            f"error={abs(observed-theory):.2e}"
        )


def plot_optimal_trajectory(
    w=8,
    r=4
):

    T = (
        20 * w + r
    )

    V_T, x_opt = solve_offline_lp(
        T=T,
        w=w
    )

    V_stationary = stationary_value(
        T
    )

    delta_T = (
        V_T - V_stationary
    )

    fig, ax = plt.subplots(
        figsize=(11, 3.8)
    )

    ax.step(
        np.arange(
            1,
            T + 1
        ),
        x_opt,
        where="mid",
        color=COLORS["blue"],
        linewidth=1.6
    )

    ax.set_xlabel(
        "Time t",
        fontsize=12
    )

    ax.set_ylabel(
        "Action value",
        fontsize=12
    )

    ax.set_title(
        f"Offline Optimal Trajectory "
        f"(w = {w}, r = {r}, "
        f"gap = {delta_T:.2f})",
        fontsize=13,
        pad=10
    )

    ax.set_ylim(
        -0.05,
        1.05
    )

    ax.grid(
        linestyle="--",
        linewidth=0.7,
        alpha=0.22
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.spines[
        "left"
    ].set_linewidth(0.8)

    ax.spines[
        "bottom"
    ].set_linewidth(0.8)

    ax.tick_params(
        axis="both",
        labelsize=10
    )

    fig.tight_layout()

    fig.savefig(
        f"imgs/"
        f"experiment1_optimal_trajectory_"
        f"w{w}_r{r}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"\nOptimal trajectory:"
        f"\nw = {w}"
        f"\nr = {r}"
        f"\nT = {T}"
        f"\nOffline value = {V_T:.4f}"
        f"\nStationary value = {V_stationary:.4f}"
        f"\nGap = {delta_T:.4f}"
    )


if __name__ == "__main__":

    results = (
        run_geometry_experiment()
    )

    plot_geometry_experiment(
        results,
        w=32
    )

    plot_optimal_trajectory(
        w=8,
        r=4
    )