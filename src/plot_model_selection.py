from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


CSV_FILE = Path(
    "/home/hadi/CRLT/llmrouting_processed/"
    "llmrouterbench_final.csv"
)

FEATURE_FILE = Path(
    "/home/hadi/CRLT/llmrouting_processed/"
    "features.npy"
)

OUTPUT_DIR = Path(
    "./llmrouter_results/figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


W = 5
BUDGET = 0.10
T = 1000
SEED = 42

LAMBDA_REG = 1.0
DELTA = 0.05
NOISE_R = 0.5
DET_FACTOR = 2.0


POLICIES = [
    "Rare-Update Optimistic Planning",
    "Myopic Feasible OFUL",
    "Unconstrained OFUL"
]


df = pd.read_csv(
    CSV_FILE
)

feature_matrix = np.load(
    FEATURE_FILE
)

assert len(df) == len(feature_matrix)


models = sorted(
    df.model.unique()
)

K = len(models)
d = feature_matrix.shape[1]


print(
    "Models:",
    models
)

print(
    "K:",
    K
)

print(
    "Feature dimension:",
    d
)


features = {}
reward_table = {}
cost_table = {}


for i, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Loading dataset"
):

    key = (
        int(row.query_id),
        row.model
    )

    features[key] = feature_matrix[i]

    reward_table[key] = float(
        row.reward
    )

    cost_table[key] = float(
        row.cost
    )


query_ids = sorted(
    df.query_id.unique()
)


if T > len(query_ids):
    raise ValueError(
        f"T={T} is larger than number of queries={len(query_ids)}"
    )


class LLMRoutingEnvironment:


    def __init__(
        self,
        query_order,
        budget
    ):

        self.query_order = query_order
        self.budget = budget
        self.history = []
        self.expected_reward = 0.0
        self.violations = 0


    def feasible_actions(
        self,
        t
    ):

        q = self.query_order[t]

        recent = self.history[
            -(W - 1):
        ]

        used_cost = sum(
            c
            for _, c in recent
        )

        actions = []

        for a, model in enumerate(models):

            cost = cost_table[
                (
                    q,
                    model
                )
            ]

            if used_cost + cost <= self.budget:
                actions.append(a)

        return actions


    def step(
        self,
        t,
        action
    ):

        q = self.query_order[t]
        model = models[action]

        reward = reward_table[
            (
                q,
                model
            )
        ]

        cost = cost_table[
            (
                q,
                model
            )
        ]

        self.history.append(
            (
                action,
                cost
            )
        )

        self.expected_reward += reward

        if len(self.history) >= W:

            window_cost = sum(
                c
                for _, c in self.history[-W:]
            )

            if window_cost > self.budget:
                self.violations += 1

        return reward


class OptimisticLinearBandit:


    def __init__(
        self
    ):

        self.V = (
            LAMBDA_REG
            *
            np.eye(d)
        )

        self.b = np.zeros(d)


    def theta_hat(
        self
    ):

        return np.linalg.solve(
            self.V,
            self.b
        )


    def beta(
        self
    ):

        return (
            NOISE_R
            *
            np.sqrt(
                d
                *
                np.log(
                    1 / DELTA
                )
            )
        )


    def score(
        self,
        x
    ):

        theta = self.theta_hat()

        uncertainty = np.sqrt(
            x
            @
            np.linalg.solve(
                self.V,
                x
            )
        )

        return (
            theta @ x
            +
            self.beta()
            *
            uncertainty
        )


    def logdet(
        self
    ):

        return np.linalg.slogdet(
            self.V
        )[1]


    def update(
        self,
        x,
        y
    ):

        self.V += np.outer(
            x,
            x
        )

        self.b += y * x


class MyopicFeasibleOFUL:


    def __init__(
        self
    ):

        self.model = OptimisticLinearBandit()
        self.n_updates = 0


    def choose(
        self,
        actions,
        q
    ):

        scores = []

        for a in actions:

            x = features[
                (
                    q,
                    models[a]
                )
            ]

            scores.append(
                self.model.score(x)
            )

        self.n_updates += 1

        return actions[
            int(np.argmax(scores))
        ]


    def update(
        self,
        x,
        y
    ):

        self.model.update(
            x,
            y
        )


class UnconstrainedOFUL(
    MyopicFeasibleOFUL
):

    pass


class RareUpdateOptimisticPlanning:


    def __init__(
        self
    ):

        self.model = OptimisticLinearBandit()
        self.frozen_scores = {}
        self.last_logdet = None
        self.n_updates = 0


    def choose(
        self,
        actions,
        q
    ):

        current_logdet = self.model.logdet()

        need_update = (
            self.last_logdet is None
            or
            current_logdet
            >
            self.last_logdet
            +
            np.log(DET_FACTOR)
        )

        if need_update:

            self.frozen_scores = {}

            for a in actions:

                x = features[
                    (
                        q,
                        models[a]
                    )
                ]

                self.frozen_scores[a] = (
                    self.model.score(x)
                )

            self.last_logdet = current_logdet
            self.n_updates += 1

        else:

            for a in actions:

                if a not in self.frozen_scores:

                    x = features[
                        (
                            q,
                            models[a]
                        )
                    ]

                    self.frozen_scores[a] = (
                        self.model.score(x)
                    )

        return max(
            actions,
            key=lambda a:
            self.frozen_scores[a]
        )


    def update(
        self,
        x,
        y
    ):

        self.model.update(
            x,
            y
        )


def run_policy(
    policy_name,
    T,
    seed,
    budget
):

    rng = np.random.default_rng(
        seed
    )

    query_order = rng.choice(
        query_ids,
        size=T,
        replace=False
    )

    env = LLMRoutingEnvironment(
        query_order,
        budget
    )

    selected_models = []

    if policy_name == "Rare-Update Optimistic Planning":

        agent = RareUpdateOptimisticPlanning()

    elif policy_name == "Myopic Feasible OFUL":

        agent = MyopicFeasibleOFUL()

    else:

        agent = UnconstrainedOFUL()

    for t in range(T):

        q = query_order[t]

        if policy_name == "Unconstrained OFUL":

            actions = list(
                range(K)
            )

        else:

            actions = env.feasible_actions(
                t
            )

        if len(actions) == 0:
            continue

        action = agent.choose(
            actions,
            q
        )

        selected_models.append(
            models[action]
        )

        reward = env.step(
            t,
            action
        )

        x = features[
            (
                q,
                models[action]
            )
        ]

        agent.update(
            x,
            reward
        )

    return {
        "expected_reward": env.expected_reward,
        "average_reward": env.expected_reward / T,
        "violations": env.violations,
        "updates": agent.n_updates,
        "selected_models": selected_models
    }


selection_frequency = {}
selection_counts = {}


for policy in POLICIES:

    print(
        f"\nRunning {policy}"
    )

    result = run_policy(
        policy,
        T,
        SEED,
        BUDGET
    )

    selected_models = result[
        "selected_models"
    ]

    if len(selected_models) == 0:
        raise RuntimeError(
            f"No selected models recorded for {policy}"
        )

    counts = Counter(
        selected_models
    )

    selection_counts[policy] = counts

    frequencies = []

    for model in models:

        freq = (
            counts[model]
            /
            len(selected_models)
            *
            100
        )

        frequencies.append(
            freq
        )

    selection_frequency[policy] = frequencies

    print(
        "Expected reward:",
        result["expected_reward"]
    )

    print(
        "Average reward:",
        result["average_reward"]
    )

    print(
        "Violations:",
        result["violations"]
    )

    print(
        "Updates:",
        result["updates"]
    )

    print(
        "Selected steps:",
        len(selected_models)
    )


print(
    "\nModel selection distribution (%)"
)

header = ["Model"] + POLICIES

print(
    "{:<35} {:>15} {:>15} {:>15}".format(
        *header
    )
)

for i, model in enumerate(models):

    values = [
        selection_frequency[p][i]
        for p in POLICIES
    ]

    print(
        "{:<35} {:>15.2f} {:>15.2f} {:>15.2f}".format(
            model,
            values[0],
            values[1],
            values[2]
        )
    )


distribution_rows = []

for policy in POLICIES:

    total_selected = sum(
        selection_counts[policy].values()
    )

    for model in models:

        count = selection_counts[policy][model]
        freq = (
            100.0 * count / total_selected
            if total_selected > 0
            else 0.0
        )

        distribution_rows.append(
            {
                "budget": BUDGET,
                "T": T,
                "seed": SEED,
                "policy": policy,
                "model": model,
                "count": count,
                "frequency_percent": freq
            }
        )

distribution_df = pd.DataFrame(
    distribution_rows
)

distribution_csv = OUTPUT_DIR / (
    f"model_selection_distribution_B{BUDGET}_T{T}.csv"
)

distribution_df.to_csv(
    distribution_csv,
    index=False
)

print(
    "\nSaved distribution table:"
)

print(
    distribution_csv
)


# =========================
# Colors
# =========================

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



# =========================
# Comparison distribution plot
# =========================

x = np.arange(
    len(models)
)

width = 0.25


plt.figure(
    figsize=(12, 5)
)


for i, policy in enumerate(POLICIES):

    bars = plt.bar(
        x + i * width,
        selection_frequency[policy],
        width,
        color=POLICY_COLORS[policy],
        label=policy
    )


    for bar, value in zip(
        bars,
        selection_frequency[policy]
    ):

        if value > 0:

            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0
            )



MODEL_LABELS = {

    "GPT-5": "GPT-5",

    "Claude-Sonnet-4": "Claude",

    "Gemini-2.5-Pro": "Gemini-Pro",

    "DeepSeek-V3": "DeepSeek",

    "Qwen3-235B": "Qwen",

    "Gemini-2.5-Flash": "Gemini-Flash"

}


plt.xticks(
    x + width,
    [MODEL_LABELS.get(m, m) for m in models],
    rotation=0,
    ha="center",
    fontsize=9
)


plt.ylabel(
    "Selection Frequency (%)"
)


plt.xlabel(
    "LLM Model"
)


plt.title(
    f"Model Selection Distribution\n"
    f"Budget={BUDGET}, Horizon={T}"
)


plt.legend(
    fontsize=8
)


plt.grid(
    False
)

# plt.grid(
#     axis="y"
# )


plt.tight_layout()



figure_file = OUTPUT_DIR / (
    f"model_selection_comparison_B{BUDGET}_T{T}.png"
)



plt.savefig(
    figure_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()



print(
    "Saved figure:"
)

print(
    figure_file
)



# =========================
# Individual policy plots
# =========================

for policy in POLICIES:


    plt.figure(
        figsize=(10, 4)
    )


    bars = plt.bar(
        models,
        selection_frequency[policy],
        color=POLICY_COLORS[policy]
    )


    plt.ylabel(
        "Selection Frequency (%)"
    )


    plt.xlabel(
        "LLM Model"
    )


    plt.title(
        f"{policy}\n"
        f"Budget={BUDGET}, Horizon={T}"
    )


    plt.xticks(
        rotation=45,
        ha="right"
    )


    plt.grid(
        axis="y"
    )


    for bar, value in zip(
        bars,
        selection_frequency[policy]
    ):


        if value > 0:

            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9
            )



    plt.tight_layout()



    per_policy_file = OUTPUT_DIR / (
        f"{policy.replace(' ', '_')}_"
        f"B{BUDGET}_T{T}.png"
    )



    plt.savefig(
        per_policy_file,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "Saved figure:"
    )

    print(
        per_policy_file
    )