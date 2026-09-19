from pathlib import Path

import numpy as np
import pandas as pd


RESULT_FILE = Path("./llmrouter_results/experiment_results_budget.csv")
TABLE_FILE = Path("./llmrouter_results/llmrouter_budget_table.tex")

SELECTED_BUDGETS = [0.15, 0.25, 0.35]
SELECTED_T = [500, 2500, 5000]

policies = [
    "Myopic Rare-Update Optimistic Planning",
    "Myopic Feasible OFUL",
    "Unconstrained OFUL",
    "Primal-Dual OFUL",
]

POLICY_NAMES = {
    "Myopic Rare-Update Optimistic Planning": "Rare-Update Optimistic Planning",
    "Myopic Feasible OFUL": "Myopic Feasible OFUL",
    "Unconstrained OFUL": "Unconstrained OFUL",
    "Primal-Dual OFUL": "Primal-Dual OFUL",
}

df = pd.read_csv(RESULT_FILE)

required_columns = [
    "budget",
    "T",
    "policy",
    "average_reward",
    "normalized_regret",
    "violations",
    "updates",
]

numeric_columns = [
    "budget",
    "T",
    "average_reward",
    "normalized_regret",
    "violations",
    "updates",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df[df["budget"].isin(SELECTED_BUDGETS)].copy()
df = df[df["T"].isin(SELECTED_T)].copy()
df = df[df["policy"].isin(policies)].copy()

df = (df.groupby(["budget", "T", "policy"], as_index=False).agg(
        mean_reward=("average_reward", "mean"),
        std_reward=("average_reward", "std"),
        n_reward=("average_reward", "count"),
        mean_regret=("normalized_regret", "mean"),
        std_regret=("normalized_regret", "std"),
        n_regret=("normalized_regret", "count"),
        mean_violations=("violations", "mean"),
        mean_updates=("updates", "mean"),
    ))

df["ci_reward"] = 1.96 * df["std_reward"] / np.sqrt(df["n_reward"])
df["ci_regret"] = 1.96 * df["std_regret"] / np.sqrt(df["n_regret"])

df["ci_reward"] = df["ci_reward"].fillna(0.0)
df["ci_regret"] = df["ci_regret"].fillna(0.0)

lines = []

lines.append(r"\begin{table*}[t]")
lines.append(r"\centering")
lines.append(r"\caption{LLM-routing performance across different sliding-window budgets. Results report mean $\pm$ 95\% confidence intervals over independent runs.}")
lines.append(r"\label{tab:llm_budget_sensitivity}")
lines.append(r"\small")
lines.append(r"\setlength{\tabcolsep}{5pt}")
lines.append(r"\renewcommand{\arraystretch}{1.1}")
lines.append(r"\begin{tabular}{cclcccc}")
lines.append(r"\toprule")
lines.append(r"Budget & $T$ & Method & Avg. Reward & Norm. Regret & Violations & Updates \\")
lines.append(r"\midrule")

for budget_idx, budget in enumerate(SELECTED_BUDGETS):
    budget_df = df[np.isclose(df["budget"], budget)]

    for T_idx, T in enumerate(SELECTED_T):
        T_df = budget_df[budget_df["T"] == T]
        first_policy = True

        for policy in policies:
            row = T_df[T_df["policy"] == policy]

            if row.empty:
                continue

            r = row.iloc[0]

            reward = f"${r['mean_reward']:.4f} \\pm {r['ci_reward']:.4f}$"
            regret = f"${r['mean_regret']:.4f} \\pm {r['ci_regret']:.4f}$"
            violations = f"{r['mean_violations']:.1f}"
            updates = f"{r['mean_updates']:.1f}"

            budget_text = f"${budget:.2f}$" if T_idx == 0 and first_policy else ""
            T_text = f"${T}$" if first_policy else ""
            method = POLICY_NAMES[policy]

            lines.append(
                f"{budget_text} & {T_text} & {method} "
                f"& {reward} & {regret} & {violations} & {updates} \\\\"
            )

            first_policy = False

        if T_idx < len(SELECTED_T) - 1:
            lines.append(r"\cmidrule(lr){2-7}")

    if budget_idx < len(SELECTED_BUDGETS) - 1:
        lines.append(r"\midrule")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table*}")

latex_table = "\n".join(lines)

print(latex_table)

TABLE_FILE.parent.mkdir(parents=True, exist_ok=True)
TABLE_FILE.write_text(latex_table)

print("Saved:", TABLE_FILE)