from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Paths
# =========================

RESULT_FILE = Path(
    "./llmrouter_results/experiment_results_budget.csv"
)


FIG_DIR = Path(
    "./llmrouter_results/figures"
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)



# =========================
# Load results
# =========================

df = pd.read_csv(
    RESULT_FILE
)


# Average over repeated runs

df = (
    df
    .groupby(
        [
            "budget",
            "T",
            "policy"
        ],
        as_index=False
    )
    .mean()
)



# =========================
# Settings
# =========================

policies = [
    "Rare-Update Optimistic Planning",
    "Myopic Feasible OFUL",
    "Unconstrained OFUL"
]


markers = {
    "Rare-Update Optimistic Planning": "o",
    "Myopic Feasible OFUL": "s",
    "Unconstrained OFUL": "^"
}


# Color palette

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

    "Rare-Update Optimistic Planning":
        COLORS["blue"],

    "Myopic Feasible OFUL":
        COLORS["orange"],

    "Unconstrained OFUL":
        COLORS["green"]

}



metrics = [
    (
        "average_reward",
        "Average Reward",
        "(a) Average Reward"
    ),

    (
        "violations",
        "Constraint Violations",
        "(b) Violations"
    ),

    (
        "updates",
        "Number of Updates",
        "(c) Updates"
    )
]



# =====================================================
# Figure 1: Main paper
# One representative budget
# =====================================================

MAIN_BUDGET = 0.10


main_df = df[
    df["budget"] == MAIN_BUDGET
]


if len(main_df) == 0:

    raise ValueError(
        f"No results found for budget {MAIN_BUDGET}"
    )



fig, axes = plt.subplots(
    1,
    3,
    figsize=(15,4)
)



for ax, (metric, ylabel, title) in zip(
    axes,
    metrics
):

    for policy in policies:


        data = main_df[
            main_df["policy"] == policy
        ].sort_values(
            "T"
        )


        ax.plot(
            data["T"],
            data[metric],
            marker=markers[policy],
            color=POLICY_COLORS[policy],
            linewidth=2,
            markersize=7,
            label=policy
        )


    ax.set_title(
        title,
        fontsize=12
    )


    ax.set_xlabel(
        "Horizon $T$",
        fontsize=11
    )


    ax.set_ylabel(
        ylabel,
        fontsize=11
    )


    ax.grid(
        True
    )



handles, labels = axes[0].get_legend_handles_labels()


fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    fontsize=9,
    bbox_to_anchor=(0.5,1.15)
)



plt.suptitle(
    f"LLM Routing Results ($B={MAIN_BUDGET}$)",
    fontsize=14,
    y=1.20
)


plt.tight_layout()



main_file = (
    FIG_DIR /
    "llmrouter_main_budget_0.10.png"
)



plt.savefig(
    main_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()



print(
    "Saved:",
    main_file
)



# =====================================================
# Figure 2: Appendix
# All budgets
# =====================================================


budgets = sorted(
    df["budget"].unique(),
    reverse=True
)



fig, axes = plt.subplots(
    3,
    len(budgets),
    figsize=(5 * len(budgets), 10),
    sharex="col"
)



if len(budgets) == 1:

    axes = axes.reshape(
        3,
        1
    )



for col, budget in enumerate(budgets):


    budget_df = df[
        df["budget"] == budget
    ]


    for row, (metric, ylabel, title) in enumerate(metrics):


        ax = axes[row, col]


        for policy in policies:


            data = budget_df[
                budget_df["policy"] == policy
            ].sort_values(
                "T"
            )


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

            ax.set_title(
                f"Budget $B={budget}$",
                fontsize=12
            )


        if col == 0:

            ax.set_ylabel(
                ylabel,
                fontsize=11
            )


        if row == 2:

            ax.set_xlabel(
                "Horizon $T$",
                fontsize=11
            )


        ax.grid(
            True
        )



handles, labels = axes[0,0].get_legend_handles_labels()


fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    fontsize=9,
    bbox_to_anchor=(0.5,1.04)
)



plt.suptitle(
    "LLM Routing Results Across Budgets",
    fontsize=14,
    y=1.08
)



plt.tight_layout()



appendix_file = (
    FIG_DIR /
    "llmrouter_all_budgets.png"
)



plt.savefig(
    appendix_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()



print(
    "Saved:",
    appendix_file
)