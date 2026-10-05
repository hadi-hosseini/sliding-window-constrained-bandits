from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


RESULT_FILE = Path("./llmrouter_results/experiment_results_budget.csv")
BUDGET = 0.4
POLICIES = ["Myopic Rare-Update Optimistic Planning", "Myopic Feasible OFUL", "Unconstrained OFUL", "Primal-Dual OFUL"]

df = pd.read_csv(RESULT_FILE)
df.columns = [column.replace("**", "").replace("\\_", "_").strip() for column in df.columns]
required_columns = ["budget", "T", "run", "policy", "average_reward", "normalized_regret", "violations", "updates"]
numeric_columns = ["budget", "T", "run", "average_reward", "normalized_regret", "violations", "updates"]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df[np.isclose(df["budget"], BUDGET)].copy()
df = df[df["policy"].isin(POLICIES)].copy()
HORIZONS = sorted(df["T"].dropna().astype(int).unique())

def mean_ci(values, confidence=0.95):
    values = pd.to_numeric(pd.Series(values), errors="coerce",).dropna().to_numpy(dtype=float)
    n = len(values)
    if n == 0:
        return np.nan, np.nan

    mean = float(np.mean(values))

    if n == 1:
        return mean, 0.0

    sem = np.std(values, ddof=1) / np.sqrt(n)
    critical_value = t.ppf(1.0 - (1.0 - confidence) / 2.0, df=n - 1)
    margin = critical_value * sem
    return mean, float(margin)


rows = []
for T_value in HORIZONS:
    for policy in POLICIES:
        data = df[(df["T"] == T_value) & (df["policy"] == policy)]

        if data.empty:
            continue

        reward_mean, reward_ci = mean_ci(data["average_reward"])
        regret_mean, regret_ci = mean_ci(data["normalized_regret"])
        violations_mean, violations_ci = mean_ci(data["violations"])
        updates_mean, updates_ci = mean_ci(data["updates"])

        rows.append(
            {
                "T": T_value,
                "Method": policy,
                "Avg Reward":
                    f"{reward_mean:.4f} ± {reward_ci:.4f}",
                "Norm Regret":
                    f"{regret_mean:.4f} ± {regret_ci:.4f}",
                "Violations":
                    f"{violations_mean:.2f} ± {violations_ci:.2f}",
                "Updates":
                    f"{updates_mean:.2f} ± {updates_ci:.2f}",
            }
        )

result = pd.DataFrame(rows)
result["Method"] = pd.Categorical(result["Method"], categories=POLICIES, ordered=True)
result = (result.sort_values(["T", "Method"]).reset_index(drop=True))
print(result.to_string(index=False))