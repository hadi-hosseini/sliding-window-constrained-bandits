from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULT_DIR = Path("./kuairand_results")
FIGURE_DIR = "/home/hadi/CRLT/imgs"
RESULT_FILE = RESULT_DIR / "experiment_results.csv"


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


POLICY_COLORS = {
    "Rare-Update Optimistic Planning": COLORS["blue"],
    "Myopic Feasible UCB": COLORS["orange"],
    "Unconstrained UCB": COLORS["red"],
    "Uniform Feasible": COLORS["green"],
}


POLICY_ORDER = [
    "Rare-Update Optimistic Planning",
    "Myopic Feasible UCB",
    "Unconstrained UCB",
    "Uniform Feasible",
]


POLICY_LABELS = {
    "Rare-Update Optimistic Planning": "Rare-Update\nPlanning",
    "Myopic Feasible UCB": "Myopic\nFeasible",
    "Unconstrained UCB": "Unconstrained",
    "Uniform Feasible": "Uniform\nFeasible",
}


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# LOAD RESULTS
# ============================================================

results = pd.read_csv(RESULT_FILE)

print("Policies found:")
print(results["policy"].value_counts())


# ============================================================
# AVERAGE REPEATED RUNS WITHIN EACH USER
#
# This makes users the independent statistical units.
# ============================================================

user_results = (
    results
    .groupby(
        ["user_id", "policy"],
        as_index=False
    )
    .agg(
        average_expected_reward=(
            "average_expected_reward",
            "mean"
        ),
        normalized_pseudo_regret=(
            "normalized_pseudo_regret",
            "mean"
        ),
        violations=(
            "violations",
            "mean"
        ),
    )
)


# ============================================================
# SUMMARY + 95% CI
# ============================================================

def summarize_metric(metric):

    stats = (
        user_results
        .groupby("policy")[metric]
        .agg(
            mean="mean",
            std="std",
            n="count",
        )
        .reset_index()
    )

    stats["se"] = (
        stats["std"]
        / np.sqrt(stats["n"])
    )

    stats["ci95"] = (
        1.96 * stats["se"]
    )

    # Keep only methods actually present
    available_order = [
        p
        for p in POLICY_ORDER
        if p in stats["policy"].values
    ]

    stats["policy"] = pd.Categorical(
        stats["policy"],
        categories=available_order,
        ordered=True,
    )

    stats = (
        stats
        .sort_values("policy")
        .reset_index(drop=True)
    )

    return stats


reward_stats = summarize_metric(
    "average_expected_reward"
)

regret_stats = summarize_metric(
    "normalized_pseudo_regret"
)

violation_stats = summarize_metric(
    "violations"
)


# ============================================================
# FUNCTION FOR ONE PANEL
# ============================================================

def draw_panel(
    ax,
    stats,
    ylabel,
    title,
    zero_line=False,
    value_format=".3f",
):

    policies = stats["policy"].astype(str).tolist()

    labels = [
        POLICY_LABELS.get(p, p)
        for p in policies
    ]

    colors = [
        POLICY_COLORS.get(
            p,
            COLORS["gray"]
        )
        for p in policies
    ]

    means = stats["mean"].to_numpy()

    errors = (
        stats["ci95"]
        .fillna(0)
        .to_numpy()
    )

    x = np.arange(
        len(policies)
    )

    bars = ax.bar(
        x,
        means,
        yerr=errors,
        capsize=4,
        width=0.68,
        color=colors,
        edgecolor=COLORS["black"],
        linewidth=0.6,
    )

    if zero_line:
        ax.axhline(
            0,
            color=COLORS["black"],
            linewidth=1.0,
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        rotation=0,
    )

    ax.set_ylabel(ylabel)

    ax.set_title(
        title,
        fontweight="bold",
    )

    ax.grid(
        axis="y",
        alpha=0.20,
        linewidth=0.7,
    )

    ax.set_axisbelow(True)

    # Put values above bars
    for bar, value in zip(
        bars,
        means
    ):

        height = bar.get_height()

        if height >= 0:
            offset = 4
            va = "bottom"
        else:
            offset = -4
            va = "top"

        ax.annotate(
            format(value, value_format),
            xy=(
                bar.get_x()
                + bar.get_width() / 2,
                height,
            ),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9,
        )


# ============================================================
# CREATE 3 PANELS SIDE BY SIDE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15.5, 4.5),
)


# ------------------------------------------------------------
# (a) Expected engagement
# ------------------------------------------------------------

draw_panel(
    axes[0],
    reward_stats,
    ylabel="Average Expected Engagement",
    title="(a) Expected Engagement",
    value_format=".3f",
)


# ------------------------------------------------------------
# (b) Normalized pseudo-regret
# ------------------------------------------------------------

draw_panel(
    axes[1],
    regret_stats,
    ylabel="Normalized Pseudo-Regret",
    title="(b) Pseudo-Regret",
    zero_line=True,
    value_format=".3f",
)


# ------------------------------------------------------------
# (c) Constraint violations
# ------------------------------------------------------------

draw_panel(
    axes[2],
    violation_stats,
    ylabel="Number of Violated Windows",
    title="(c) Constraint Violations",
    value_format=".1f",
)


fig.tight_layout(w_pad=2.5)
png_file = (FIGURE_DIR / "kuairand_comparison.png")


fig.savefig(png_file, bbox_inches="tight")

print("Saved:", png_file)
plt.show()