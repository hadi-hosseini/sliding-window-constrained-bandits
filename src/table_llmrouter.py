
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import t


# =========================
# Path
# =========================

RESULT_FILE = Path(
    "./llmrouter_results/experiment_results_budget.csv"
)


# =========================
# Settings
# =========================

BUDGET = 0.10

HORIZONS = [
    500,
    1000
]

POLICIES = [
    "Rare-Update Optimistic Planning",
    "Myopic Feasible OFUL",
    "Unconstrained OFUL"
]


# =========================
# Load results
# =========================

df = pd.read_csv(
    RESULT_FILE
)


df = df[
    df["budget"] == BUDGET
]


df = df[
    df["T"].isin(HORIZONS)
]


df = df[
    df["policy"].isin(POLICIES)
]


# =========================
# Confidence interval
# =========================

def mean_ci(values, confidence=0.95):

    values = np.array(
        values
    )

    n = len(values)

    mean = np.mean(
        values
    )

    if n <= 1:
        return mean, 0.0


    sem = (
        np.std(
            values,
            ddof=1
        )
        /
        np.sqrt(n)
    )


    margin = (
        t.ppf(
            (1 + confidence) / 2,
            n - 1
        )
        *
        sem
    )


    return mean, margin



# =========================
# Generate table
# =========================

rows = []


for T_value in HORIZONS:

    for policy in POLICIES:


        data = df[
            (df["T"] == T_value)
            &
            (df["policy"] == policy)
        ]


        reward_mean, reward_ci = mean_ci(
            data["average_reward"]
        )


        regret_mean, regret_ci = mean_ci(
            data["normalized_regret"]
        )


        violations_mean = data[
            "violations"
        ].mean()


        updates_mean = data[
            "updates"
        ].mean()



        rows.append(
            {
                "T": T_value,
                "Method": policy,
                "Avg Reward": 
                    f"{reward_mean:.4f} ± {reward_ci:.4f}",

                "Norm Regret":
                    f"{regret_mean:.4f} ± {regret_ci:.4f}",

                "Violations":
                    f"{violations_mean:.1f}",

                "Updates":
                    f"{updates_mean:.1f}"
            }
        )



result = pd.DataFrame(
    rows
)


print(
    result.to_string(
        index=False
    )
)