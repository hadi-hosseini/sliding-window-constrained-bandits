from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULT_FILE = Path("./llmrouter_results/experiment_results_budget.csv")
FIG_DIR = Path("./imgs")
FIG_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(RESULT_FILE)
required_columns = ["budget", "T", "policy", "average_reward", "normalized_regret", "violations", "updates"]
numeric_columns = ["budget", "T", "average_reward", "normalized_regret", "violations", "updates"]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

SELECTED_BUDGETS = [0.15, 0.25, 0.35]
df = df[df["budget"].isin(SELECTED_BUDGETS)].copy()
df = (df.groupby(["budget", "T", "policy"], as_index=False).agg(
        average_reward=("average_reward", "mean"),
        normalized_regret=("normalized_regret", "mean"),
        violations=("violations", "mean"),
        updates=("updates", "mean"),
    ))

policies = ["Rare-Update Optimistic Planning", "Myopic Feasible OFUL", "Unconstrained OFUL", "Primal-Dual OFUL"]
markers = {"Rare-Update Optimistic Planning": "o", "Myopic Feasible OFUL": "s", "Unconstrained OFUL": "^", "Primal-Dual OFUL": "D"}
COLORS = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00"}
POLICY_COLORS = {"Rare-Update Optimistic Planning": COLORS["blue"], "Myopic Feasible OFUL": COLORS["orange"], "Unconstrained OFUL": COLORS["green"], "Primal-Dual OFUL": COLORS["red"]}

metrics = [("average_reward", "Average Reward"), ("normalized_regret", "Normalized Pseudo-Regret"), ("violations", "Constraint Violations"), ("updates", "Number of Updates"),]
budgets = SELECTED_BUDGETS
fig, axes = plt.subplots(len(budgets), len(metrics), figsize=(20, 10), sharex=False)

if len(budgets) == 1:
    axes = axes.reshape(1, len(metrics))

for row, budget in enumerate(budgets):
    budget_df = df[df["budget"] == budget]

    for col, (metric, title) in enumerate(metrics):
        ax = axes[row, col]
        for policy in policies:
            data = (budget_df[budget_df["policy"] == policy].sort_values("T"))

            if data.empty:
                continue

            ax.plot(
                data["T"],
                data[metric],
                marker=markers[policy],
                color=POLICY_COLORS[policy],
                linewidth=2,
                markersize=6,
                label=policy,
            )

        if row == 0:
            ax.set_title(title, fontsize=12)

        if col == 0:
            ax.set_ylabel(f"Budget $B={budget:g}$", fontsize=11)

        ax.set_xlabel("Horizon $T$", fontsize=10,)
        ax.grid(True, alpha=0.3)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=11, frameon=False,)
plt.tight_layout(rect=[0, 0, 1, 0.93])
appendix_file = (FIG_DIR / "llmrouter_budget_sensitivity.png")
plt.savefig(appendix_file, dpi=300, bbox_inches="tight",)
plt.close()
print("Saved:", appendix_file,)