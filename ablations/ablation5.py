import os
import itertools
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# Setup
# ============================================================

os.makedirs("imgs", exist_ok=True)

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


# ============================================================
# Non-cyclic planning instance
# ============================================================

# Actions:
# 0  = (0,0)
# e1 = (1,0)
# e2 = (0,1)
ACTIONS = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
])

ACTION_NAMES = [
    "0",
    "e1",
    "e2",
]

D = 2
W = 4


# Position-dependent sliding-window constraint:
#
# 4 c^T x_1 + 2 c^T x_2 + c^T x_3 + c^T x_4 <= 14
#
COST_VECTOR = np.array([
    1.0,
    3.0,
])

POSITION_WEIGHTS = np.array([
    4.0,
    2.0,
    1.0,
    1.0,
])

BUDGET = 14.0

ACTION_COSTS = (
    ACTIONS @ COST_VECTOR
)


# ============================================================
# Reward model
# ============================================================

THETA_STAR = np.array([
    0.6,
    1.6,
])

TRUE_REWARDS = (
    ACTIONS @ THETA_STAR
)


# ============================================================
# Experiment parameters
# ============================================================

T = 8192

ALPHA_VALUES = np.array([
    1.25,
    1.50,
    2.00,
    4.00,
    8.00,
])

NOISE_STD = 0.1

LAMBDA_REG = 1.0
DELTA = 0.05

L_BOUND = np.max(
    np.linalg.norm(
        ACTIONS,
        axis=1
    )
)

S_BOUND = np.linalg.norm(
    THETA_STAR
)

REWARD_BOUND = (
    L_BOUND
    *
    S_BOUND
)

N_RUNS = 30
BASE_SEED = 2026


# ============================================================
# History-state model
# ============================================================

def build_state_model():

    # Partial histories of lengths 0,...,w-1.
    states = [()]

    for length in range(
        1,
        W
    ):
        states.extend(
            itertools.product(
                range(
                    len(ACTIONS)
                ),
                repeat=length
            )
        )

    states = [
        tuple(state)
        for state in states
    ]

    state_to_index = {
        state: index
        for index, state in enumerate(
            states
        )
    }

    n_states = len(
        states
    )

    n_actions = len(
        ACTIONS
    )

    next_state = -np.ones(
        (
            n_states,
            n_actions
        ),
        dtype=int
    )

    for state_index, state in enumerate(
        states
    ):

        for action_index in range(
            n_actions
        ):

            # No complete window exists yet.
            if len(state) < W - 1:

                is_feasible = True

            else:

                window = (
                    state
                    +
                    (
                        action_index,
                    )
                )

                costs = np.array([
                    ACTION_COSTS[index]
                    for index in window
                ])

                weighted_cost = np.dot(
                    POSITION_WEIGHTS,
                    costs
                )

                is_feasible = (
                    weighted_cost
                    <=
                    BUDGET + 1e-12
                )

            if is_feasible:

                new_state = (
                    state
                    +
                    (
                        action_index,
                    )
                )

                new_state = (
                    new_state[
                        -(W - 1):
                    ]
                )

                next_state[
                    state_index,
                    action_index
                ] = (
                    state_to_index[
                        new_state
                    ]
                )

    # Infeasible transitions are mapped to one extra
    # state whose Bellman value is -infinity.
    safe_next_state = (
        next_state.copy()
    )

    safe_next_state[
        safe_next_state < 0
    ] = n_states

    return {
        "states":
            states,
        "state_to_index":
            state_to_index,
        "next_state":
            next_state,
        "safe_next_state":
            safe_next_state,
        "n_states":
            n_states,
        "empty_state":
            state_to_index[()],
    }


STATE_MODEL = (
    build_state_model()
)


# ============================================================
# Confidence sets
# ============================================================

def confidence_radius(
    V,
    lambda_reg=LAMBDA_REG,
    noise_scale=NOISE_STD,
    theta_bound=S_BOUND,
    delta=DELTA,
):

    d = V.shape[0]

    sign, logdet_v = (
        np.linalg.slogdet(V)
    )

    if sign <= 0:
        raise ValueError(
            "V must be positive definite."
        )

    logdet_initial = (
        d
        *
        np.log(
            lambda_reg
        )
    )

    log_term = (
        0.5
        *
        (
            logdet_v
            -
            logdet_initial
        )
        +
        np.log(
            1.0 / delta
        )
    )

    beta = (
        noise_scale
        *
        np.sqrt(
            max(
                0.0,
                2.0 * log_term
            )
        )
        +
        np.sqrt(
            lambda_reg
        )
        *
        theta_bound
    )

    return float(
        beta
    )


def optimistic_rewards(
    V,
    b,
):

    theta_hat = (
        np.linalg.solve(
            V,
            b
        )
    )

    beta = (
        confidence_radius(
            V=V
        )
    )

    V_inv = np.linalg.inv(
        V
    )

    optimistic = np.zeros(
        len(ACTIONS)
    )

    for action_index, action in enumerate(
        ACTIONS
    ):

        bonus = (
            beta
            *
            np.sqrt(
                max(
                    0.0,
                    action
                    @ V_inv
                    @ action
                )
            )
        )

        optimistic[
            action_index
        ] = (
            theta_hat
            @ action
            +
            bonus
        )

    # Clipped optimistic reward from the theory.
    optimistic = np.clip(
        optimistic,
        -REWARD_BOUND,
        REWARD_BOUND
    )

    return optimistic


# ============================================================
# Exact remaining-horizon Bellman planner
# ============================================================

def bellman_table(
    stage_rewards,
    horizon,
):

    n_states = (
        STATE_MODEL[
            "n_states"
        ]
    )

    safe_next_state = (
        STATE_MODEL[
            "safe_next_state"
        ]
    )

    values = np.full(
        (
            horizon + 1,
            n_states
        ),
        -np.inf
    )

    policy = np.full(
        (
            horizon + 1,
            n_states
        ),
        -1,
        dtype=np.int16
    )

    values[
        0,
        :
    ] = 0.0

    for n in range(
        1,
        horizon + 1
    ):

        previous = np.concatenate([
            values[
                n - 1
            ],
            np.array([
                -np.inf
            ])
        ])

        q_values = (
            stage_rewards[
                None,
                :
            ]
            +
            previous[
                safe_next_state
            ]
        )

        best_values = (
            np.max(
                q_values,
                axis=1
            )
        )

        best_actions = (
            np.argmax(
                q_values,
                axis=1
            )
        )

        viable = (
            np.isfinite(
                best_values
            )
        )

        values[
            n
        ] = (
            best_values
        )

        policy[
            n,
            viable
        ] = (
            best_actions[
                viable
            ]
        )

    return (
        values,
        policy
    )


# ============================================================
# Exact offline benchmark
# ============================================================

@lru_cache(maxsize=None)
def offline_optimum(
    horizon
):

    values, _ = (
        bellman_table(
            stage_rewards=TRUE_REWARDS,
            horizon=int(
                horizon
            )
        )
    )

    initial_state = (
        STATE_MODEL[
            "empty_state"
        ]
    )

    return float(
        values[
            int(horizon),
            initial_state
        ]
    )


# ============================================================
# Constraint checks
# ============================================================

def count_window_violations(
    action_indices,
    tolerance=1e-10,
):

    if len(
        action_indices
    ) < W:
        return 0

    violations = 0

    for t in range(
        W - 1,
        len(
            action_indices
        )
    ):

        window = (
            action_indices[
                t - W + 1:
                t + 1
            ]
        )

        costs = np.array([
            ACTION_COSTS[
                action_index
            ]
            for action_index in window
        ])

        weighted_cost = np.dot(
            POSITION_WEIGHTS,
            costs
        )

        if (
            weighted_cost
            >
            BUDGET + tolerance
        ):
            violations += 1

    return int(
        violations
    )


def assert_feasible(
    action_indices
):

    violations = (
        count_window_violations(
            action_indices
        )
    )

    if violations != 0:

        raise RuntimeError(
            f"Generated trajectory contains "
            f"{violations} sliding-window violations."
        )


# ============================================================
# Rare-update optimistic planning with threshold alpha
# ============================================================

def run_planning_with_threshold(
    noise,
    alpha,
):

    horizon = len(
        noise
    )

    if alpha <= 1.0:
        raise ValueError(
            "alpha must be greater than 1."
        )

    V = (
        LAMBDA_REG
        *
        np.eye(D)
    )

    b = np.zeros(
        D
    )

    state = (
        STATE_MODEL[
            "empty_state"
        ]
    )

    next_state = (
        STATE_MODEL[
            "next_state"
        ]
    )

    action_indices = []

    total_true_reward = 0.0

    t = 0
    n_updates = 0

    while t < horizon:

        # ----------------------------------------------------
        # Start a new episode and freeze statistics.
        # ----------------------------------------------------

        _, episode_logdet = (
            np.linalg.slogdet(V)
        )

        optimistic = (
            optimistic_rewards(
                V=V,
                b=b
            )
        )

        remaining_at_start = (
            horizon - t
        )

        # Exact optimistic remaining-horizon plan.
        _, policy = (
            bellman_table(
                stage_rewards=optimistic,
                horizon=remaining_at_start
            )
        )

        n_updates += 1

        # ----------------------------------------------------
        # Follow the frozen plan until
        #
        # det(V_t) > alpha det(V_s_k).
        # ----------------------------------------------------

        while t < horizon:

            _, current_logdet = (
                np.linalg.slogdet(V)
            )

            threshold_reached = (
                current_logdet
                >
                episode_logdet
                +
                np.log(
                    alpha
                )
                +
                1e-12
            )

            if threshold_reached:
                break

            remaining = (
                horizon - t
            )

            action_index = int(
                policy[
                    remaining,
                    state
                ]
            )

            if action_index < 0:

                raise RuntimeError(
                    f"Planner reached a non-viable "
                    f"state at t={t}, alpha={alpha}."
                )

            successor = (
                next_state[
                    state,
                    action_index
                ]
            )

            if successor < 0:

                raise RuntimeError(
                    "Planner selected an infeasible action."
                )

            action = (
                ACTIONS[
                    action_index
                ]
            )

            action_indices.append(
                action_index
            )

            true_reward = (
                THETA_STAR
                @ action
            )

            observation = (
                true_reward
                +
                noise[
                    t
                ]
            )

            total_true_reward += (
                true_reward
            )

            # Statistical observations are retained.
            V += np.outer(
                action,
                action
            )

            b += (
                action
                *
                observation
            )

            state = (
                successor
            )

            t += 1

    # Every generated trajectory must be exactly feasible.
    assert_feasible(
        action_indices
    )

    benchmark = (
        offline_optimum(
            horizon
        )
    )

    regret = (
        benchmark
        -
        total_true_reward
    )

    # Remove tiny numerical error only.
    if (
        regret < 0
        and
        abs(regret) < 1e-8
    ):
        regret = 0.0

    return {
        "normalized_regret":
            float(
                regret / horizon
            ),
        "updates":
            int(
                n_updates
            ),
    }


# ============================================================
# Problem summary
# ============================================================

def print_problem_summary():

    print(
        "\nPlanning update-threshold ablation"
    )

    print(
        "-" * 70
    )

    print(
        f"Horizon T = {T}"
    )

    print(
        f"Window length w = {W}"
    )

    print(
        "Threshold values: "
        f"{list(ALPHA_VALUES)}"
    )

    print(
        f"Number of runs = {N_RUNS}"
    )

    benchmark = (
        offline_optimum(
            T
        )
    )

    print(
        f"Offline optimum = "
        f"{benchmark:.4f}"
    )

    print(
        f"Offline average reward = "
        f"{benchmark / T:.4f}"
    )


# ============================================================
# Run ablation
# ============================================================

def run_planning_threshold_ablation(
    n_runs=N_RUNS,
    seed=BASE_SEED,
):

    results = {
        "normalized_regret":
            np.zeros(
                (
                    len(
                        ALPHA_VALUES
                    ),
                    n_runs
                )
            ),
        "updates":
            np.zeros(
                (
                    len(
                        ALPHA_VALUES
                    ),
                    n_runs
                )
            ),
    }

    for run in tqdm(
        range(n_runs),
        desc="Runs"
    ):

        rng = (
            np.random.default_rng(
                seed + run
            )
        )

        # Use the same noise realization for
        # every alpha within this run.
        noise = (
            rng.normal(
                loc=0.0,
                scale=NOISE_STD,
                size=T
            )
        )

        for alpha_index, alpha in enumerate(
            ALPHA_VALUES
        ):

            result = (
                run_planning_with_threshold(
                    noise=noise,
                    alpha=float(
                        alpha
                    )
                )
            )

            results[
                "normalized_regret"
            ][
                alpha_index,
                run
            ] = (
                result[
                    "normalized_regret"
                ]
            )

            results[
                "updates"
            ][
                alpha_index,
                run
            ] = (
                result[
                    "updates"
                ]
            )

    return results


# ============================================================
# Statistics
# ============================================================

def mean_and_ci(
    values
):

    mean = np.mean(
        values,
        axis=1
    )

    if values.shape[1] == 1:

        ci = np.zeros_like(
            mean
        )

    else:

        standard_error = (
            np.std(
                values,
                axis=1,
                ddof=1
            )
            /
            np.sqrt(
                values.shape[1]
            )
        )

        ci = (
            1.96
            *
            standard_error
        )

    return (
        mean,
        ci
    )


def print_results(
    results
):

    regret_mean, regret_ci = (
        mean_and_ci(
            results[
                "normalized_regret"
            ]
        )
    )

    update_mean, update_ci = (
        mean_and_ci(
            results[
                "updates"
            ]
        )
    )

    print(
        "\nPlanning threshold results"
    )

    print(
        "-" * 90
    )

    for index, alpha in enumerate(
        ALPHA_VALUES
    ):

        print(
            f"alpha={alpha:4.2f} | "
            f"regret/T "
            f"{regret_mean[index]:8.5f} "
            f"± {regret_ci[index]:7.5f} | "
            f"updates "
            f"{update_mean[index]:6.2f} "
            f"± {update_ci[index]:5.2f}"
        )


# ============================================================
# Plot
# ============================================================

def plot_planning_threshold_ablation(
    results
):

    regret_mean, regret_ci = (
        mean_and_ci(
            results[
                "normalized_regret"
            ]
        )
    )

    update_mean, update_ci = (
        mean_and_ci(
            results[
                "updates"
            ]
        )
    )

    # Categorical spacing makes all five
    # threshold values equally easy to read.
    x = np.arange(
        len(
            ALPHA_VALUES
        )
    )

    alpha_labels = [
        "1.25",
        "1.5",
        "2",
        "4",
        "8",
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.2)
    )


    # --------------------------------------------------------
    # Left: normalized pseudo-regret
    # --------------------------------------------------------

    ax1 = axes[0]

    ax1.plot(
        x,
        regret_mean,
        color=COLORS["blue"],
        marker="o",
        markersize=7,
        linewidth=2.2,
        markeredgecolor="white",
        markeredgewidth=0.7,
        zorder=3
    )

    ax1.fill_between(
        x,
        regret_mean - regret_ci,
        regret_mean + regret_ci,
        color=COLORS["blue"],
        alpha=0.30
    )

    ax1.set_title(
        "(a) Normalized Pseudo-Regret",
        fontsize=14,
        pad=12
    )

    ax1.set_xlabel(
        "Update threshold α",
        fontsize=13
    )

    ax1.set_ylabel(
        "Pseudo-regret / T",
        fontsize=13
    )

    ax1.set_xticks(
        x
    )

    ax1.set_xticklabels(
        alpha_labels
    )

    ax1.tick_params(
        axis="both",
        labelsize=11
    )

    ax1.grid(
        linestyle="--",
        linewidth=0.7,
        alpha=0.22
    )

    ax1.spines[
        "top"
    ].set_visible(False)

    ax1.spines[
        "right"
    ].set_visible(False)

    ax1.spines[
        "left"
    ].set_linewidth(0.8)

    ax1.spines[
        "bottom"
    ].set_linewidth(0.8)

    ax1.set_ylim(
        bottom=0
    )


    # --------------------------------------------------------
    # Right: number of planning updates
    # --------------------------------------------------------

    ax2 = axes[1]

    ax2.plot(
        x,
        update_mean,
        color=COLORS["orange"],
        marker="s",
        markersize=7,
        linewidth=2.2,
        markeredgecolor="white",
        markeredgewidth=0.7,
        zorder=3
    )

    ax2.fill_between(
        x,
        update_mean - update_ci,
        update_mean + update_ci,
        color=COLORS["orange"],
        alpha=0.30
    )

    ax2.set_title(
        "(b) Policy Updates",
        fontsize=14,
        pad=12
    )

    ax2.set_xlabel(
        "Update threshold α",
        fontsize=13
    )

    ax2.set_ylabel(
        "Number of policy updates",
        fontsize=13
    )

    ax2.set_xticks(
        x
    )

    ax2.set_xticklabels(
        alpha_labels
    )

    ax2.tick_params(
        axis="both",
        labelsize=11
    )

    ax2.grid(
        linestyle="--",
        linewidth=0.7,
        alpha=0.22
    )

    ax2.spines[
        "top"
    ].set_visible(False)

    ax2.spines[
        "right"
    ].set_visible(False)

    ax2.spines[
        "left"
    ].set_linewidth(0.8)

    ax2.spines[
        "bottom"
    ].set_linewidth(0.8)

    ax2.set_ylim(
        bottom=0
    )


    # --------------------------------------------------------
    # Highlight the default alpha = 2
    # --------------------------------------------------------

    default_index = int(
        np.where(
            np.isclose(
                ALPHA_VALUES,
                2.0
            )
        )[0][0]
    )

    ax1.axvline(
        default_index,
        color=COLORS["gray"],
        linestyle="--",
        linewidth=1.2,
        alpha=0.65
    )

    ax2.axvline(
        default_index,
        color=COLORS["gray"],
        linestyle="--",
        linewidth=1.2,
        alpha=0.65
    )


    # --------------------------------------------------------
    # Layout and save
    # --------------------------------------------------------

    fig.tight_layout(
        w_pad=3.0
    )

    fig.savefig(
        "imgs/ablation5.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print_problem_summary()

    results = (
        run_planning_threshold_ablation(
            n_runs=N_RUNS,
            seed=BASE_SEED
        )
    )

    print_results(
        results
    )

    plot_planning_threshold_ablation(
        results
    )