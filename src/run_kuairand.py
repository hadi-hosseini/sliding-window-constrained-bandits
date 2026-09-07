import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PLOT EXPERIMENT RESULTS
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


# ------------------------------------------------------------
# Load summary results
# ------------------------------------------------------------

summary_file = OUTPUT_DIR / "experiment_summary.csv"
df = pd.read_csv(summary_file)


# Keep a fixed policy order
policy_order = [
    "Myopic Feasible UCB",
    "Unconstrained UCB",
    "Uniform Feasible",
]

df = (
    df.set_index("policy")
      .loc[policy_order]
      .reset_index()
)


# ------------------------------------------------------------
# Colors for policies
# ------------------------------------------------------------

policy_colors = {
    "Myopic Feasible UCB": COLORS["blue"],
    "Unconstrained UCB": COLORS["orange"],
    "Uniform Feasible": COLORS["green"],
}

bar_colors = [
    policy_colors[p]
    for p in df["policy"]
]


# ------------------------------------------------------------
# Create figure
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(14, 4.5),
)


# ============================================================
# 1. EXPECTED REWARD
# ============================================================

axes[0].bar(
    df["policy"],
    df["mean_expected_reward"],
    yerr=df["std_expected_reward"],
    color=bar_colors,
    edgecolor=COLORS["black"],
    linewidth=0.8,
    capsize=5,
)

axes[0].set_ylabel("Average Expected Reward")
axes[0].set_title("(a) Expected Reward")

axes[0].grid(
    axis="y",
    linestyle="--",
    alpha=0.3,
)

axes[0].set_axisbelow(True)


# ============================================================
# 2. NORMALIZED REGRET
# ============================================================

axes[1].bar(
    df["policy"],
    df["mean_normalized_regret"],
    yerr=df["std_normalized_regret"],
    color=bar_colors,
    edgecolor=COLORS["black"],
    linewidth=0.8,
    capsize=5,
)

axes[1].set_ylabel("Normalized Pseudo-Regret")
axes[1].set_title("(b) Normalized Regret")

axes[1].grid(
    axis="y",
    linestyle="--",
    alpha=0.3,
)

axes[1].set_axisbelow(True)


# ============================================================
# 3. CONSTRAINT VIOLATIONS
# ============================================================

axes[2].bar(
    df["policy"],
    df["mean_violations"],
    yerr=df["std_violations"],
    color=bar_colors,
    edgecolor=COLORS["black"],
    linewidth=0.8,
    capsize=5,
)

axes[2].set_ylabel("Number of Violations")
axes[2].set_title("(c) Constraint Violations")

axes[2].grid(
    axis="y",
    linestyle="--",
    alpha=0.3,
)

axes[2].set_axisbelow(True)


# ------------------------------------------------------------
# Clean x-axis labels
# ------------------------------------------------------------

short_labels = [
    "Myopic\nFeasible UCB",
    "Unconstrained\nUCB",
    "Uniform\nFeasible",
]

for ax in axes:
    ax.set_xticks(
        np.arange(len(short_labels))
    )

    ax.set_xticklabels(
        short_labels,
        fontsize=9,
    )

    # Remove top/right borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

plt.tight_layout()


# ------------------------------------------------------------
# Save publication-quality versions
# ------------------------------------------------------------

figure_png = OUTPUT_DIR / "policy_comparison.png"
figure_pdf = OUTPUT_DIR / "policy_comparison.pdf"

plt.savefig(
    figure_png,
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    figure_pdf,
    bbox_inches="tight",
)

plt.show()

print("Saved:", figure_png)
print("Saved:", figure_pdf)