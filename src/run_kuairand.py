from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("./kuairand_processed")
OUTPUT_DIR = Path("./kuairand_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXPERIMENTAL PARAMETERS
# ============================================================

W = 20
C = 5

T_VALUES = [
    500,
    1000,
    1500,
    2000,
    2500,
    3000,
    3500,
    4000,
    4500,
    5000,
]

N_USERS = 100
N_RUNS = 10

MIN_CATEGORY_INTERACTIONS = 50

THETA_COLUMN = "theta_hat"
RANDOM_SEED = 42


# ============================================================
# POLICIES
# ============================================================

policies = [
    "Rare-Update Optimistic Planning",
    "Myopic Feasible OFUL",
    "Unconstrained OFUL",
]


# ============================================================
# LOAD DATA
# ============================================================

top_categories = pd.read_csv(
    DATA_DIR / "top_categories.csv"
)

preferences = pd.read_csv(
    DATA_DIR / "user_category_preferences.csv"
)

coverage = pd.read_csv(
    DATA_DIR / "user_coverage.csv"
)


# Number of actions
K = len(top_categories)


# ============================================================
# OFUL PARAMETERS
# ============================================================

LAMBDA_REG = 1.0
DELTA = 0.05

# Bernoulli centered noise is 1/2-sub-Gaussian
NOISE_R = 0.5

# Since theta_j in [0,1]
THETA_NORM_BOUND = np.sqrt(K)

# determinant-doubling threshold
DET_FACTOR = 2.0


# ============================================================
# ELIGIBLE USERS
# ============================================================

eligible = coverage[
    (coverage["num_categories"] == K)
    &
    (
        coverage["min_category_interactions"]
        >= MIN_CATEGORY_INTERACTIONS
    )
].copy()


rng_users = np.random.default_rng(
    RANDOM_SEED
)


if (
    N_USERS is not None
    and len(eligible) > N_USERS
):

    selected_user_ids = rng_users.choice(
        eligible["user_id"].to_numpy(),
        size=N_USERS,
        replace=False,
    )

else:

    selected_user_ids = (
        eligible["user_id"].to_numpy()
    )


# ============================================================
# USER-SPECIFIC TRUE REWARD VECTOR
# ============================================================

def get_theta(user_id):

    user_data = preferences[
        preferences["user_id"] == user_id
    ].copy()

    user_data = user_data.sort_values(
        "action_id"
    )


    if len(user_data) != K:
        raise ValueError(
            f"User {user_id} has {len(user_data)} "
            f"categories; expected {K}."
        )


    action_ids = (
        user_data["action_id"]
        .astype(int)
        .to_numpy()
    )

    if not np.array_equal(
        action_ids,
        np.arange(K),
    ):
        raise ValueError(
            f"Unexpected action IDs for user {user_id}: "
            f"{action_ids}"
        )


    theta_star = (
        user_data[THETA_COLUMN]
        .astype(float)
        .to_numpy()
    )

    return theta_star


# ============================================================
# OFFLINE-OPTIMAL FEASIBLE VALUE
# ============================================================

def offline_value(
    theta_star,
    T,
    w=20,
    c=5,
):

    if w % c != 0:
        raise ValueError(
            "offline_value requires w divisible by c."
        )

    if T % w != 0:
        raise ValueError(
            "offline_value currently assumes T divisible by w."
        )


    number_of_categories = (
        w // c
    )

    best = (
        np.sort(theta_star)[::-1]
        [:number_of_categories]
    )

    value_per_block = (
        c * best.sum()
    )

    return (
        (T // w)
        * value_per_block
    )


# ============================================================
# ENVIRONMENT
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
            dtype=float,
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

        recent_history = (
            self.history[
                -(self.w - 1):
            ]
        )

        counts = Counter(
            recent_history
        )

        feasible = [
            action
            for action in range(self.K)
            if counts[action] < self.c
        ]

        return feasible


    def step(
        self,
        action,
    ):

        action = int(action)

        expected_reward = float(
            self.theta[action]
        )

        reward = int(
            self.rng.binomial(
                1,
                expected_reward,
            )
        )


        self.history.append(
            action
        )


        violation = False

        if len(self.history) >= self.w:

            window = (
                self.history[-self.w:]
            )

            counts = Counter(
                window
            )

            violation = any(
                count > self.c
                for count
                in counts.values()
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
# OFUL ESTIMATOR
# ============================================================

class OptimisticLinearBandit:

    def __init__(
        self,
        K,
        lambda_reg=1.0,
        delta=0.05,
        noise_r=0.5,
        theta_norm_bound=None,
    ):

        self.K = K

        self.lambda_reg = lambda_reg
        self.delta = delta
        self.noise_r = noise_r


        if theta_norm_bound is None:
            theta_norm_bound = np.sqrt(K)


        self.theta_norm_bound = (
            theta_norm_bound
        )


        # Since actions are one-hot,
        # V_t is diagonal.

        self.v_diag = np.full(
            K,
            lambda_reg,
            dtype=float,
        )


        # b_t = sum x_q Y_q

        self.b = np.zeros(
            K,
            dtype=float,
        )


    def theta_hat(self):

        return (
            self.b
            / self.v_diag
        )


    def logdet(self):

        return float(
            np.log(
                self.v_diag
            ).sum()
        )


    def beta(self):

        logdet_v = (
            self.logdet()
        )

        logdet_lambda = (
            self.K
            * np.log(
                self.lambda_reg
            )
        )


        log_term = (
            0.5
            * (
                logdet_v
                - logdet_lambda
            )
            +
            np.log(
                1.0
                / self.delta
            )
        )


        beta = (
            self.noise_r
            * np.sqrt(
                2.0
                * max(
                    log_term,
                    0.0,
                )
            )
            +
            np.sqrt(
                self.lambda_reg
            )
            * self.theta_norm_bound
        )


        return float(beta)


    def optimistic_scores(self):

        theta_hat = (
            self.theta_hat()
        )

        beta = (
            self.beta()
        )


        # For one-hot e_j:
        #
        # ||e_j||_{V^{-1}}
        # =
        # 1 / sqrt(V_jj)

        bonus = (
            beta
            / np.sqrt(
                self.v_diag
            )
        )


        scores = (
            theta_hat
            + bonus
        )


        return np.clip(
            scores,
            0.0,
            1.0,
        )


    def update(
        self,
        action,
        reward,
    ):

        action = int(action)

        self.v_diag[action] += 1.0

        self.b[action] += reward


# ============================================================
# MYOPIC / UNCONSTRAINED OFUL
# ============================================================

class OFULPolicy:

    def __init__(
        self,
        K,
        lambda_reg=1.0,
        delta=0.05,
        noise_r=0.5,
        theta_norm_bound=None,
        seed=0,
    ):

        self.K = K

        self.rng = np.random.default_rng(
            seed
        )


        self.model = OptimisticLinearBandit(
            K=K,
            lambda_reg=lambda_reg,
            delta=delta,
            noise_r=noise_r,
            theta_norm_bound=theta_norm_bound,
        )


        # These policies recompute their
        # optimistic decision every round.

        self.n_updates = 0


    def choose(
        self,
        actions,
    ):

        actions = np.asarray(
            actions,
            dtype=int,
        )


        if len(actions) == 0:
            raise RuntimeError(
                "No available action."
            )


        scores = (
            self.model
            .optimistic_scores()[
                actions
            ]
        )


        max_score = (
            scores.max()
        )


        candidates = (
            actions[
                np.isclose(
                    scores,
                    max_score,
                )
            ]
        )


        self.n_updates += 1


        return int(
            self.rng.choice(
                candidates
            )
        )


    def update(
        self,
        action,
        reward,
    ):

        self.model.update(
            action,
            reward,
        )


# ============================================================
# RARE-UPDATE OPTIMISTIC PLANNING
# ============================================================

class RareUpdateOptimisticPlanning:

    def __init__(
        self,
        K,
        w,
        c,
        lambda_reg=1.0,
        delta=0.05,
        noise_r=0.5,
        theta_norm_bound=None,
        det_factor=2.0,
        seed=0,
    ):

        self.K = K
        self.w = w
        self.c = c

        self.det_factor = (
            det_factor
        )


        self.rng = (
            np.random.default_rng(
                seed
            )
        )


        self.model = OptimisticLinearBandit(
            K=K,
            lambda_reg=lambda_reg,
            delta=delta,
            noise_r=noise_r,
            theta_norm_bound=theta_norm_bound,
        )


        # log det(W_k)
        self.episode_logdet = None

        # U_k held fixed inside an episode
        self.frozen_scores = None

        # Number of replanning events
        self.n_updates = 0


        # Feasible initialization
        self.initial_order = (
            self.rng.permutation(
                K
            )
        )


    def start_new_episode(self):

        # W_k = V_{t_k}
        self.episode_logdet = (
            self.model.logdet()
        )


        # Freeze optimistic objective U_k
        self.frozen_scores = (
            self.model
            .optimistic_scores()
        )


        self.n_updates += 1


    def determinant_doubled(self):

        if self.episode_logdet is None:
            return True


        return (
            self.model.logdet()
            >
            self.episode_logdet
            + np.log(
                self.det_factor
            )
        )


    def initialization_action(
        self,
        t,
        feasible_actions,
    ):

        index = (
            (t - 1)
            % self.K
        )


        preferred = int(
            self.initial_order[
                index
            ]
        )


        if preferred in feasible_actions:
            return preferred


        return int(
            self.rng.choice(
                feasible_actions
            )
        )


    def planning_action(
        self,
        feasible_actions,
    ):

        feasible_actions = np.asarray(
            feasible_actions,
            dtype=int,
        )


        scores = (
            self.frozen_scores[
                feasible_actions
            ]
        )


        max_score = (
            scores.max()
        )


        candidates = (
            feasible_actions[
                np.isclose(
                    scores,
                    max_score,
                )
            ]
        )


        return int(
            self.rng.choice(
                candidates
            )
        )


    def choose(
        self,
        feasible_actions,
        t,
    ):

        feasible_actions = list(
            feasible_actions
        )


        if len(feasible_actions) == 0:
            raise RuntimeError(
                "No feasible action available."
            )


        # Feasible initialization prefix
        if t < self.w:

            return (
                self.initialization_action(
                    t=t,
                    feasible_actions=feasible_actions,
                )
            )


        # Determinant-doubling update
        if self.determinant_doubled():

            self.start_new_episode()


        return (
            self.planning_action(
                feasible_actions
            )
        )


    def update(
        self,
        action,
        reward,
    ):

        self.model.update(
            action,
            reward,
        )


# ============================================================
# RUN ONE POLICY
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


    # --------------------------------------------------------
    # Proposed method
    # --------------------------------------------------------

    if (
        policy_name
        == "Rare-Update Optimistic Planning"
    ):

        agent = RareUpdateOptimisticPlanning(
            K=len(theta_star),
            w=W,
            c=C,
            lambda_reg=LAMBDA_REG,
            delta=DELTA,
            noise_r=NOISE_R,
            theta_norm_bound=THETA_NORM_BOUND,
            det_factor=DET_FACTOR,
            seed=seed + 1000,
        )


    # --------------------------------------------------------
    # Myopic Feasible OFUL
    # --------------------------------------------------------

    elif (
        policy_name
        == "Myopic Feasible OFUL"
    ):

        agent = OFULPolicy(
            K=len(theta_star),
            lambda_reg=LAMBDA_REG,
            delta=DELTA,
            noise_r=NOISE_R,
            theta_norm_bound=THETA_NORM_BOUND,
            seed=seed + 1000,
        )


    # --------------------------------------------------------
    # Unconstrained OFUL
    # --------------------------------------------------------

    elif (
        policy_name
        == "Unconstrained OFUL"
    ):

        agent = OFULPolicy(
            K=len(theta_star),
            lambda_reg=LAMBDA_REG,
            delta=DELTA,
            noise_r=NOISE_R,
            theta_norm_bound=THETA_NORM_BOUND,
            seed=seed + 1000,
        )


    else:

        raise ValueError(
            f"Unknown policy: {policy_name}"
        )


    # ========================================================
    # INTERACTION
    # ========================================================

    for t in range(
        1,
        T + 1,
    ):

        feasible = (
            env.feasible_actions()
        )


        if (
            policy_name
            == "Rare-Update Optimistic Planning"
        ):

            action = agent.choose(
                feasible_actions=feasible,
                t=t,
            )


        elif (
            policy_name
            == "Myopic Feasible OFUL"
        ):

            action = agent.choose(
                feasible
            )


        elif (
            policy_name
            == "Unconstrained OFUL"
        ):

            action = agent.choose(
                np.arange(
                    len(theta_star)
                )
            )


        reward, _, _ = (
            env.step(
                action
            )
        )


        agent.update(
            action,
            reward,
        )


    # ========================================================
    # EVALUATION
    # ========================================================

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

        "updates":
            agent.n_updates,
    }


# ============================================================
# COMPLETE EXPERIMENT
# ============================================================

def run():

    results = []


    for user_index, user_id in enumerate(
        tqdm(
            selected_user_ids,
            desc="Processing users",
            unit="user",
        )
    ):

        theta_star = get_theta(
            user_id
        )


        for T in T_VALUES:

            for run_idx in range(
                N_RUNS
            ):

                for policy_name in policies:

                    seed = (
                        RANDOM_SEED
                        + 100000 * user_index
                        + 100 * run_idx
                    )


                    outcome = run_policy(
                        theta_star=theta_star,
                        policy_name=policy_name,
                        T=T,
                        seed=seed,
                    )


                    results.append({
                        "user_id": user_id,
                        "run_idx": run_idx,
                        "T": T,
                        "policy": policy_name,
                        **outcome,
                    })


    # ========================================================
    # RAW RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )


    results_df[
        "average_offline_reward"
    ] = (
        results_df["offline_value"]
        / results_df["T"]
    )


    results_file = (
        OUTPUT_DIR
        / "experiment_results.csv"
    )


    results_df.to_csv(
        results_file,
        index=False,
    )


    # ========================================================
    # FIRST AVERAGE RUNS WITHIN EACH USER
    # ========================================================

    user_results = (
        results_df
        .groupby(
            [
                "user_id",
                "T",
                "policy",
            ],
            as_index=False,
        )
        .agg(

            offline_value=(
                "offline_value",
                "mean",
            ),

            average_offline_reward=(
                "average_offline_reward",
                "mean",
            ),

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


    # ========================================================
    # THEN AVERAGE ACROSS USERS
    # ========================================================

    summary_results = (
        user_results
        .groupby(
            [
                "T",
                "policy",
            ],
            as_index=False,
        )
        .agg(

            mean_offline_value=(
                "offline_value",
                "mean",
            ),

            std_offline_value=(
                "offline_value",
                "std",
            ),

            mean_average_offline_reward=(
                "average_offline_reward",
                "mean",
            ),

            std_average_offline_reward=(
                "average_offline_reward",
                "std",
            ),

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

            mean_updates=(
                "updates",
                "mean",
            ),

            std_updates=(
                "updates",
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
        index=False,
    )


    print(
        summary_results.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    run()