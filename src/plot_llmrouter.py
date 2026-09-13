from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULT_FILE = Path("./llmrouter_results/experiment_results_budget.csv")
FIG_DIR = Path("./llmrouter_results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULT_FILE)
df = (df.groupby(["budget", "T", "policy"], as_index=False).mean())


MAIN_BUDGET = 0.10
policies = ["Rare-Update Optimistic Planning", "Myopic Feasible OFUL", "Unconstrained OFUL"]
markers = {"Rare-Update Optimistic Planning": "o", "Myopic Feasible OFUL": "s", "Unconstrained OFUL": "^"}
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
    "Unconstrained OFUL": COLORS["green"]
}

metrics = [
    (
        "average_reward",
        "Average Reward",
        "(a) Average Reward"
    ),
    (
        "normalized_regret",
        "Normalized Pseudo-Regret",
        "(b) Normalized Pseudo-Regret"
    ),
    (
        "violations",
        "Constraint Violations",
        "(c) Violations"
    ),
    (
        "updates",
        "Number of Updates",
        "(d) Updates"
    )
]


budgets = sorted(df["budget"].unique(), reverse=True)
fig, axes = plt.subplots(4, len(budgets), figsize=(5 * len(budgets), 13), sharex="col")

if len(budgets) == 1:
    axes = axes.reshape(4, 1)

for col, budget in enumerate(budgets):
    budget_df = df[df["budget"] == budget]

    for row, (metric, ylabel, title) in enumerate(metrics):
        ax = axes[row, col]

        for policy in policies:
            data = budget_df[budget_df["policy"] == policy].sort_values("T")

            ax.plot(
                data["T"],
                data[metric],
                marker=markers[policy],
                color=POLICY_COLORS[policy],
                linewidth=2,
                markersize=6,
                label=policy
            )

        if row == 0:
            ax.set_title(f"Budget $B={budget}$", fontsize=12)

        if col == 0:
            ax.set_ylabel(ylabel, fontsize=11)

        if row == 3:
            ax.set_xlabel("Horizon $T$", fontsize=11)

        ax.grid(True)

handles, labels = axes[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    fontsize=15,
)

plt.tight_layout(rect=[0, 0, 1, 0.88])
appendix_file = (FIG_DIR / "llmrouter_all_budgets.png")
plt.savefig(appendix_file, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", appendix_file)