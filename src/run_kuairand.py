from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd


DATA_DIR = Path("./kuairand_processed")
OUTPUT_DIR = Path("./kuairand_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sliding-window constraint
W = 20
C = 5

# Horizon. Keep divisible by W for the simple offline benchmark.
T = 5000

# Number of repeated simulations for each user
N_RUNS = 10

# Minimum observations required for every selected category
MIN_CATEGORY_INTERACTIONS = 50

# None = use every eligible user
# Set, e.g., 100 for faster experiments.
N_USERS = 4

# Which preference estimate to treat as theta*
# Alternatives: "raw_click_rate", "theta_hat"
THETA_COLUMN = "theta_hat"

RANDOM_SEED = 42


# ============================================================
# 1. LOAD THE PROCESSED CSV FILES
# ============================================================

summary = pd.read_csv(
    DATA_DIR / "dataset_summary.csv"
)

categories = pd.read_csv(
    DATA_DIR / "all_first_level_categories.csv"
)

top_categories = pd.read_csv(
    DATA_DIR / "top_categories.csv"
)

preferences = pd.read_csv(
    DATA_DIR / "user_category_preferences.csv"
)

coverage = pd.read_csv(
    DATA_DIR / "user_coverage.csv"
)


print("\n================ DATASET ================")
print(summary)

print("\n================ TOP CATEGORIES ================")
print(
    top_categories[
        [
            "action_id",
            "category_id",
            "category_name",
            "n_interactions",
            "click_rate",
        ]
    ].to_string(index=False)
)


# ============================================================
# 2. DETERMINE K
# ============================================================

K = len(top_categories)

print("\nNumber of actions K =", K)

# Make sure action IDs are 0,...,K-1
top_categories = top_categories.sort_values(
    "action_id"
).reset_index(drop=True)

expected_ids = np.arange(K)

actual_ids = top_categories[
    "action_id"
].astype(int).to_numpy()

if not np.array_equal(
    actual_ids,
    expected_ids
):
    raise ValueError(
        "action_id must be 0,...,K-1."
    )


# ============================================================
# 3. SELECT RELIABLE USERS
# ============================================================

eligible = coverage[
    (coverage["num_categories"] == K)
    &
    (
        coverage["min_category_interactions"]
        >= MIN_CATEGORY_INTERACTIONS
    )
].copy()


print(
    "\nEligible users:",
    len(eligible),
    "/",
    len(coverage)
)


# Optional reproducible subsampling
rng_users = np.random.default_rng(
    RANDOM_SEED
)

if N_USERS is not None and len(eligible) > N_USERS:

    selected_user_ids = rng_users.choice(
        eligible["user_id"].to_numpy(),
        size=N_USERS,
        replace=False,
    )

else:

    selected_user_ids = eligible[
        "user_id"
    ].to_numpy()


print(
    "Users used in experiment:",
    len(selected_user_ids)
)


# ============================================================
# 4. CONSTRUCT theta*_u FOR A USER
# ============================================================

def get_theta(user_id):

    user_data = preferences[
        preferences["user_id"] == user_id
    ].copy()

    user_data = user_data.sort_values(
        "action_id"
    )

    # Ensure all K actions exist
    if len(user_data) != K:
        raise ValueError(
            f"User {user_id} does not have "
            f"all {K} categories."
        )

    action_ids = (
        user_data["action_id"]
        .astype(int)
        .to_numpy()
    )

    if not np.array_equal(
        action_ids,
        np.arange(K)
    ):
        raise ValueError(
            f"Invalid action mapping for user {user_id}"
        )

    theta_star = (
        user_data[THETA_COLUMN]
        .astype(float)
        .to_numpy()
    )

    return theta_star


# Example
example_user = selected_user_ids[0]

theta_example = get_theta(
    example_user
)

print(
    "\nExample user:",
    example_user
)

print(
    "theta* =",
    np.round(theta_example, 4)
)


# ============================================================
# 5. EXACT SLIDING-WINDOW ENVIRONMENT
# ============================================================

class RollingExposureEnvironment:

    def __init__(
        self,
        theta_star,
        w=20,
        c=5,
        seed=0,
    ):

        self.theta = np.asarray(
            theta_star,
            dtype=float
        )

        self.K = len(self.theta)

        self.w = w
        self.c = c

        self.rng = np.random.default_rng(
            seed
        )

        self.history = []

        self.realized_reward = 0
        self.expected_reward = 0.0
        self.n_violations = 0


    def feasible_actions(self):

        # Only previous w-1 actions matter
        recent_history = self.history[
            -(self.w - 1):
        ]

        counts = Counter(
            recent_history
        )

        feasible = [
            a
            for a in range(self.K)
            if counts[a] < self.c
        ]

        return feasible


    def step(self, action):

        action = int(action)

        expected_reward = float(
            self.theta[action]
        )

        reward = int(
            self.rng.binomial(
                1,
                expected_reward
            )
        )

        self.history.append(action)

        # Count violations exactly as in the paper:
        # one violation for each complete window
        # containing any over-exposed category.
        violation = False

        if len(self.history) >= self.w:

            window = self.history[-self.w:]

            counts = Counter(window)

            violation = any(
                count > self.c
                for count in counts.values()
            )

            if violation:
                self.n_violations += 1


        self.realized_reward += reward

        self.expected_reward += (
            expected_reward
        )

        return (
            reward,
            expected_reward,
            violation,
        )


# ============================================================
# 6. UCB POLICY
# ============================================================

class UCB:

    def __init__(
        self,
        K,
        seed=0,
    ):

        self.K = K

        self.counts = np.zeros(
            K,
            dtype=int
        )

        self.reward_sum = np.zeros(
            K,
            dtype=float
        )

        self.rng = np.random.default_rng(
            seed
        )


    def choose(
        self,
        actions,
        t,
    ):

        actions = np.asarray(
            actions,
            dtype=int
        )

        # Explore actions not yet tried
        unseen = actions[
            self.counts[actions] == 0
        ]

        if len(unseen) > 0:

            return int(
                self.rng.choice(unseen)
            )


        empirical_mean = (
            self.reward_sum[actions]
            /
            self.counts[actions]
        )

        bonus = np.sqrt(
            2.0
            * np.log(max(t, 2))
            /
            self.counts[actions]
        )

        scores = (
            empirical_mean
            + bonus
        )

        max_score = scores.max()

        candidates = actions[
            np.isclose(
                scores,
                max_score
            )
        ]

        return int(
            self.rng.choice(candidates)
        )


    def update(
        self,
        action,
        reward,
    ):

        self.counts[action] += 1

        self.reward_sum[action] += reward


# ============================================================
# 7. UNIFORM FEASIBLE POLICY
# ============================================================

class UniformFeasible:

    def __init__(
        self,
        seed=0,
    ):

        self.rng = np.random.default_rng(
            seed
        )


    def choose(
        self,
        feasible_actions,
    ):

        return int(
            self.rng.choice(
                feasible_actions
            )
        )


# ============================================================
# 8. OFFLINE FEASIBLE VALUE
# ============================================================

def offline_value(
    theta_star,
    T,
    w=20,
    c=5,
):

    """
    Exact for the experiment considered here when:
        T is divisible by w,
        rewards are nonnegative,
        and the rolling cap is c per category.

    For w=20 and c=5, each 20-round period
    uses the four highest-reward categories
    exactly five times.
    """

    if T % w != 0:
        raise ValueError(
            "Use T divisible by w for "
            "this benchmark."
        )

    if w % c != 0:
        raise ValueError(
            "This simple formula assumes "
            "w is divisible by c."
        )

    number_of_categories = (
        w // c
    )

    best = np.sort(
        theta_star
    )[::-1][
        :number_of_categories
    ]

    value_per_block = (
        c * best.sum()
    )

    return (
        T // w
    ) * value_per_block


# ============================================================
# 9. RUN ONE POLICY
# ============================================================

def run_policy(
    theta_star,
    policy_name,
    T,
    seed,
):

    env = RollingExposureEnvironment(
        theta_star=theta_star,
        w=W,
        c=C,
        seed=seed,
    )


    if policy_name == "Myopic Feasible UCB":

        agent = UCB(
            K=len(theta_star),
            seed=seed + 1000,
        )


    elif policy_name == "Unconstrained UCB":

        agent = UCB(
            K=len(theta_star),
            seed=seed + 1000,
        )


    elif policy_name == "Uniform Feasible":

        agent = UniformFeasible(
            seed=seed + 1000
        )


    else:

        raise ValueError(
            f"Unknown policy: {policy_name}"
        )


    for t in range(1, T + 1):

        feasible = (
            env.feasible_actions()
        )


        # ------------------------------------------
        # Myopic Feasible UCB
        # ------------------------------------------

        if policy_name == "Myopic Feasible UCB":

            action = agent.choose(
                feasible,
                t,
            )


        # ------------------------------------------
        # Unconstrained UCB
        #
        # Deliberately ignores feasibility.
        # ------------------------------------------

        elif policy_name == "Unconstrained UCB":

            action = agent.choose(
                np.arange(
                    len(theta_star)
                ),
                t,
            )


        # ------------------------------------------
        # Uniform Feasible
        # ------------------------------------------

        elif policy_name == "Uniform Feasible":

            action = agent.choose(
                feasible
            )


        reward, expected, violation = (
            env.step(action)
        )


        if policy_name in {
            "Myopic Feasible UCB",
            "Unconstrained UCB",
        }:

            agent.update(
                action,
                reward,
            )


    V_off = offline_value(
        theta_star=theta_star,
        T=T,
        w=W,
        c=C,
    )


    pseudo_regret = (
        V_off
        - env.expected_reward
    )


    return {
        "realized_reward":
            env.realized_reward,

        "average_realized_reward":
            env.realized_reward / T,

        "expected_reward":
            env.expected_reward,

        "average_expected_reward":
            env.expected_reward / T,

        "offline_value":
            V_off,

        "pseudo_regret":
            pseudo_regret,

        "normalized_pseudo_regret":
            pseudo_regret / T,

        "violations":
            env.n_violations,
    }


# ============================================================
# 10. RUN EXPERIMENT OVER USERS
# ============================================================

policies = [
    "Myopic Feasible UCB",
    "Unconstrained UCB",
    "Uniform Feasible",
]


results = []


for user_index, user_id in enumerate(
    selected_user_ids
):

    theta_star = get_theta(
        user_id
    )


    if user_index % 10 == 0:

        print(
            f"Processing user "
            f"{user_index + 1}/"
            f"{len(selected_user_ids)}"
        )


    for run in range(N_RUNS):

        for policy_name in policies:

            seed = (
                RANDOM_SEED
                + 100000 * user_index
                + 100 * run
            )

            outcome = run_policy(
                theta_star=theta_star,
                policy_name=policy_name,
                T=T,
                seed=seed,
            )


            results.append({
                "user_id":
                    user_id,

                "run":
                    run,

                "policy":
                    policy_name,

                **outcome,
            })


# ============================================================
# 11. SAVE RAW RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_file = (
    OUTPUT_DIR
    / "experiment_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

print(
    "\nSaved:",
    results_file
)


# ============================================================
# 12. CREATE SUMMARY TABLE
# ============================================================

summary_results = (
    results_df
    .groupby(
        "policy",
        as_index=False
    )
    .agg(
        mean_expected_reward=(
            "average_expected_reward",
            "mean",
        ),

        std_expected_reward=(
            "average_expected_reward",
            "std",
        ),

        mean_normalized_regret=(
            "normalized_pseudo_regret",
            "mean",
        ),

        std_normalized_regret=(
            "normalized_pseudo_regret",
            "std",
        ),

        mean_violations=(
            "violations",
            "mean",
        ),

        std_violations=(
            "violations",
            "std",
        ),
    )
)


summary_file = (
    OUTPUT_DIR
    / "experiment_summary.csv"
)

summary_results.to_csv(
    summary_file,
    index=False
)


print(
    "\n================ RESULTS ================"
)

print(
    summary_results.to_string(
        index=False
    )
)

print(
    "\nSaved:",
    summary_file
)