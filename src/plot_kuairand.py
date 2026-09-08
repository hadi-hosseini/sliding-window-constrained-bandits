from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as student_t


# ============================================================
# PATHS
# ============================================================

RESULT_DIR = Path("./kuairand_results")
FIGURE_DIR = Path("./imgs")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILE = RESULT_DIR / "experiment_results.csv"


# ============================================================
# COLORS
# ============================================================

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
    "Myopic Feasible OFUL": COLORS["orange"],
    "Unconstrained OFUL": COLORS["red"],
}


POLICY_LINESTYLES = {
    "Rare-Update Optimistic Planning": "-",
    "Myopic Feasible OFUL": "--",
    "Unconstrained OFUL": "-.",
}


POLICY_ORDER = [
    "Rare-Update Optimistic Planning",
    "Myopic Feasible OFUL",
    "Unconstrained OFUL",
]


POLICY_LABELS = {
    "Rare-Update Optimistic Planning":
        "Rare-Update Planning",

    "Myopic Feasible OFUL":
        "Myopic Feasible OFUL",

    "Unconstrained OFUL":
        "Unconstrained OFUL",
}


# ============================================================
# PLOT SETTINGS
# ============================================================

CONFIDENCE_LEVEL = 0.95

LINE_WIDTH = 2.3

# Transparency of error bars
CI_ALPHA = 0.45


plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
})


# ============================================================
# LOAD RESULTS
# ============================================================

results = pd.read_csv(
    RESULT_FILE
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = {
    "user_id",
    "T",
    "policy",
    "average_expected_reward",
    "normalized_pseudo_regret",
    "violations",
    "updates",
}


missing_columns = (
    required_columns
    - set(results.columns)
)


if missing_columns:
    raise ValueError(
        f"Missing columns in {RESULT_FILE}: "
        f"{sorted(missing_columns)}"
    )


# ============================================================
# OFFLINE REWARD
# ============================================================

if "average_offline_reward" not in results.columns:

    if "offline_value" not in results.columns:
        raise ValueError(
            "The results file must contain either "
            "'average_offline_reward' or 'offline_value'."
        )

    results["average_offline_reward"] = (
        results["offline_value"]
        / results["T"]
    )


# ============================================================
# CHECK AVAILABLE POLICIES
# ============================================================

available_policies = [
    policy
    for policy in POLICY_ORDER
    if policy in results["policy"].unique()
]


if not available_policies:
    raise ValueError(
        "None of the expected policies were found. "
        f"Found policies: {sorted(results['policy'].unique())}"
    )


print("Policies found:")

for policy in available_policies:
    print(" -", policy)


# ============================================================
# FIRST AVERAGE REPEATED RUNS WITHIN EACH USER
#
# After this step, each observation represents one
# (user, T, policy).
# ============================================================

user_results = (
    results
    .groupby(
        [
            "user_id",
            "T",
            "policy",
        ],
        as_index=False,
    )
    .agg(
        average_expected_reward=(
            "average_expected_reward",
            "mean",
        ),

        normalized_pseudo_regret=(
            "normalized_pseudo_regret",
            "mean",
        ),

        violations=(
            "violations",
            "mean",
        ),

        updates=(
            "updates",
            "mean",
        ),
    )
)


# ============================================================
# 95% STUDENT-t CONFIDENCE INTERVAL
# ============================================================

def ci_half_width(
    std,
    n,
    confidence=0.95,
):

    if n <= 1 or pd.isna(std):
        return 0.0

    alpha = (
        1.0
        - confidence
    )

    critical_value = student_t.ppf(
        1.0 - alpha / 2.0,
        df=n - 1,
    )

    return (
        critical_value
        * std
        / np.sqrt(n)
    )


# ============================================================
# SUMMARY ACROSS USERS
# ============================================================

summary = (
    user_results
    .groupby(
        [
            "T",
            "policy",
        ],
        as_index=False,
    )
    .agg(

        # ----------------------------------------------------
        # REWARD
        # ----------------------------------------------------

        mean_reward=(
            "average_expected_reward",
            "mean",
        ),

        std_reward=(
            "average_expected_reward",
            "std",
        ),

        n_reward=(
            "average_expected_reward",
            "count",
        ),


        # ----------------------------------------------------
        # NORMALIZED REGRET
        # ----------------------------------------------------

        mean_normalized_regret=(
            "normalized_pseudo_regret",
            "mean",
        ),

        std_normalized_regret=(
            "normalized_pseudo_regret",
            "std",
        ),

        n_normalized_regret=(
            "normalized_pseudo_regret",
            "count",
        ),


        # ----------------------------------------------------
        # VIOLATIONS
        # ----------------------------------------------------

        mean_violations=(
            "violations",
            "mean",
        ),


        # ----------------------------------------------------
        # POLICY UPDATES
        # ----------------------------------------------------

        mean_updates=(
            "updates",
            "mean",
        ),
    )
)


# ============================================================
# COMPUTE 95% CONFIDENCE INTERVALS
#
# We use confidence intervals only for:
#   - reward
#   - normalized pseudo-regret
#
# We do NOT shade violations or policy updates.
# ============================================================

summary["ci_reward"] = summary.apply(
    lambda row:
        ci_half_width(
            std=row["std_reward"],
            n=int(row["n_reward"]),
            confidence=CONFIDENCE_LEVEL,
        ),
    axis=1,
)


summary["ci_normalized_regret"] = summary.apply(
    lambda row:
        ci_half_width(
            std=row["std_normalized_regret"],
            n=int(row["n_normalized_regret"]),
            confidence=CONFIDENCE_LEVEL,
        ),
    axis=1,
)


# ============================================================
# OFFLINE BENCHMARK
#
# The offline optimum depends on the user but not on
# the stochastic reward realization or policy.
#
# We plot only its mean across users.
# No confidence region is shown.
# ============================================================

offline_user = (
    results[
        [
            "user_id",
            "T",
            "average_offline_reward",
        ]
    ]
    .drop_duplicates(
        subset=[
            "user_id",
            "T",
        ]
    )
)


offline_summary = (
    offline_user
    .groupby(
        "T",
        as_index=False,
    )
    .agg(
        mean_offline_reward=(
            "average_offline_reward",
            "mean",
        ),
    )
)


# ============================================================
# DRAW PANEL
# ============================================================

def draw_panel(
    ax,
    metric,
    ylabel,
    title,
    ci_metric=None,
    zero_line=False,
):

    for policy in available_policies:

        policy_data = (
            summary[
                summary["policy"] == policy
            ]
            .sort_values("T")
        )


        x = (
            policy_data["T"]
            .to_numpy()
        )

        y = (
            policy_data[metric]
            .to_numpy()
        )


        color = (
            POLICY_COLORS[policy]
        )

        linestyle = (
            POLICY_LINESTYLES[policy]
        )


        # ----------------------------------------------------
        # MEAN CURVE
        # ----------------------------------------------------

        ax.plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=LINE_WIDTH,
            label=POLICY_LABELS[policy],
            zorder=3,
        )


        # ----------------------------------------------------
        # 95% CONFIDENCE INTERVAL
        #
        # Error bars only.
        # No markers and no shaded bands.
        # ----------------------------------------------------

        if ci_metric is not None:

            ci = (
                policy_data[ci_metric]
                .to_numpy()
            )

            ax.errorbar(
                x,
                y,
                yerr=ci,
                fmt="none",
                ecolor=color,
                elinewidth=0.9,
                capsize=2.0,
                capthick=0.9,
                alpha=CI_ALPHA,
                zorder=2,
            )


    # --------------------------------------------------------
    # ZERO REFERENCE LINE
    # --------------------------------------------------------

    if zero_line:

        ax.axhline(
            y=0,
            color=COLORS["gray"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            zorder=1,
        )


    # --------------------------------------------------------
    # AXIS STYLE
    # --------------------------------------------------------

    ax.set_xlabel(
        "Horizon $T$"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title,
        fontweight="semibold",
        pad=10,
    )


    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.55,
        alpha=0.20,
    )


    ax.set_axisbelow(
        True
    )

    ax.margins(
        x=0.02
    )


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(18.5, 4.7),
)


# ============================================================
# (a) AVERAGE EXPECTED REWARD
# ============================================================

draw_panel(
    axes[0],
    metric="mean_reward",
    ci_metric="ci_reward",
    ylabel="Average Expected Reward",
    title="(a) Average Expected Reward",
)


# ============================================================
# ADD OFFLINE-OPTIMAL BENCHMARK
#
# Black dashed line only.
# No confidence interval.
# ============================================================

offline_data = (
    offline_summary
    .sort_values("T")
)


axes[0].plot(
    offline_data["T"],
    offline_data["mean_offline_reward"],
    color=COLORS["black"],
    linestyle=(0, (5, 3)),
    linewidth=2.0,
    label="Offline Optimal",
    zorder=4,
)


# ============================================================
# (b) NORMALIZED PSEUDO-REGRET
# ============================================================

draw_panel(
    axes[1],
    metric="mean_normalized_regret",
    ci_metric="ci_normalized_regret",
    ylabel="Normalized Pseudo-Regret",
    title="(b) Normalized Pseudo-Regret",
    zero_line=True,
)


# ============================================================
# (c) CONSTRAINT VIOLATIONS
#
# No confidence intervals.
# ============================================================

draw_panel(
    axes[2],
    metric="mean_violations",
    ylabel="Number of Violated Windows",
    title="(c) Constraint Violations",
)


axes[2].set_ylim(
    bottom=0
)


# ============================================================
# (d) POLICY UPDATES
#
# No confidence intervals.
# ============================================================

draw_panel(
    axes[3],
    metric="mean_updates",
    ylabel="Number of Policy Updates",
    title="(d) Policy Updates",
)


axes[3].set_ylim(
    bottom=0
)


# ============================================================
# X AXIS
# ============================================================

T_VALUES = sorted(
    summary["T"].unique()
)


for ax in axes:

    ax.set_xticks(
        T_VALUES
    )

    ax.set_xticklabels(
        [
            f"{int(T):,}"
            for T in T_VALUES
        ],
        rotation=35,
        ha="right",
    )


# ============================================================
# LEGEND
# ============================================================

handles, labels = (
    axes[0]
    .get_legend_handles_labels()
)


fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        1.02,
    ),
    ncol=len(labels),
    frameon=False,
    columnspacing=1.8,
    handlelength=2.8,
)


# ============================================================
# LAYOUT
# ============================================================

fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.90,
    ],
    w_pad=2.0,
)


# ============================================================
# SAVE
# ============================================================

png_file = (
    FIGURE_DIR
    / "kuairand_comparison_vs_T.png"
)

pdf_file = (
    FIGURE_DIR
    / "kuairand_comparison_vs_T.pdf"
)


fig.savefig(
    png_file,
    bbox_inches="tight",
)

fig.savefig(
    pdf_file,
    bbox_inches="tight",
)


print(
    "Saved:",
    png_file
)

print(
    "Saved:",
    pdf_file
)


plt.show()