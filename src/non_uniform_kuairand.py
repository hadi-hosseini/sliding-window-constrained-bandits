# ============================================================
# non_uniform_kuairand.py
#
# Ablation:
# Non-uniform Diversity Constraints
#
# Compare:
#   1. Rare-Update Optimistic Planning
#   2. Myopic Feasible OFUL
#
# Constraint:
#   sum_{tau=t-w+1}^{t} I(a_tau=j) <= C_j
# ============================================================


from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# Import original KuaiRand utilities
# ============================================================

from run_kuairand import (
    OptimisticLinearBandit,
    get_theta,
    selected_user_ids,
    K,
    W,
    LAMBDA_REG,
    DELTA,
    NOISE_R,
    THETA_NORM_BOUND,
    DET_FACTOR,
    RANDOM_SEED,
)



# ============================================================
# Settings
# ============================================================

OUTPUT_DIR = Path(
    "./kuairand_results_non_uniform"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


T_VALUES = [
    500,
    2500,
    5000
]


N_RUNS = 5


POLICIES = [
    "Rare-Update Optimistic Planning",
    "Myopic Feasible OFUL"
]



# ============================================================
# Non-uniform constraints C_j
# ============================================================

def create_non_uniform_constraints(K):

    C_j = np.zeros(
        K,
        dtype=int
    )


    for j in range(K):

        if j % 3 == 0:

            C_j[j] = 3

        elif j % 3 == 1:

            C_j[j] = 5

        else:

            C_j[j] = 7


    return C_j



C_j = create_non_uniform_constraints(
    K
)


print(
    "Non-uniform constraints:"
)

print(C_j)



# ============================================================
# Environment
# ============================================================

class NonUniformEnvironment:


    def __init__(
        self,
        theta_star,
        C_j,
        w,
        seed
    ):

        self.theta = np.asarray(
            theta_star
        )

        self.K = len(
            theta_star
        )

        self.C_j = C_j

        self.w = w

        self.rng = np.random.default_rng(
            seed
        )


        self.history = []

        self.expected_reward = 0.0

        self.realized_reward = 0

        self.violations = 0



    def feasible_actions(self):

        recent = self.history[
            -(self.w-1):
        ]


        counts = Counter(
            recent
        )


        feasible = []


        for a in range(self.K):

            if counts[a] < self.C_j[a]:

                feasible.append(a)


        return feasible



    def step(
        self,
        action
    ):


        expected = float(
            self.theta[action]
        )


        reward = int(
            self.rng.binomial(
                1,
                expected
            )
        )


        self.history.append(
            action
        )


        if len(self.history) >= self.w:


            window = self.history[
                -self.w:
            ]


            counts = Counter(
                window
            )


            for a,count in counts.items():

                if count > self.C_j[a]:

                    self.violations += 1

                    break



        self.expected_reward += expected

        self.realized_reward += reward


        return reward



# ============================================================
# Policies
# ============================================================

class MyopicFeasibleOFUL:


    def __init__(self):

        self.model = OptimisticLinearBandit()

        self.n_updates = 0



    def choose(
        self,
        feasible_actions
    ):

        scores = []


        for a in feasible_actions:

            scores.append(
                self.model.optimistic_scores()[a]
            )


        self.n_updates += 1


        return feasible_actions[
            np.argmax(scores)
        ]



    def update(
        self,
        action,
        reward
    ):

        self.model.update(
            action,
            reward
        )



class RareUpdatePlanning:


    def __init__(self):

        self.model = OptimisticLinearBandit()

        self.last_logdet = None

        self.frozen_scores = {}

        self.n_updates = 0



    def choose(
        self,
        feasible_actions
    ):


        current_logdet = (
            self.model.logdet()
        )


        update = (
            self.last_logdet is None
            or
            current_logdet >
            self.last_logdet + np.log(DET_FACTOR)
        )


        if update:


            self.frozen_scores = {}


            scores = (
                self.model.optimistic_scores()
            )


            for a in feasible_actions:

                self.frozen_scores[a] = scores[a]


            self.last_logdet = current_logdet

            self.n_updates += 1



        else:

            scores = (
                self.model.optimistic_scores()
            )


            for a in feasible_actions:

                if a not in self.frozen_scores:

                    self.frozen_scores[a] = scores[a]



        return max(
            feasible_actions,
            key=lambda a:
            self.frozen_scores[a]
        )



    def update(
        self,
        action,
        reward
    ):

        self.model.update(
            action,
            reward
        )



# ============================================================
# Run one experiment
# ============================================================

def run_policy(
    theta_star,
    policy,
    T,
    seed
):


    env = NonUniformEnvironment(
        theta_star,
        C_j,
        W,
        seed
    )



    if policy == "Rare-Update Optimistic Planning":

        agent = RareUpdatePlanning()


    elif policy == "Myopic Feasible OFUL":

        agent = MyopicFeasibleOFUL()



    for t in range(T):


        feasible = (
            env.feasible_actions()
        )


        if len(feasible)==0:

            feasible = list(
                range(K)
            )



        action = agent.choose(
            feasible
        )


        reward = env.step(
            action
        )


        agent.update(
            action,
            reward
        )



    return {


        "average_reward":
            env.expected_reward / T,


        "violations":
            env.violations,


        "updates":
            agent.n_updates

    }



# ============================================================
# Main
# ============================================================

def run():

    results = []


    for user_id in tqdm(
        selected_user_ids,
        desc="Users"
    ):


        theta_star = get_theta(
            user_id
        )


        for T in T_VALUES:


            for run_id in range(
                N_RUNS
            ):


                seed = (
                    RANDOM_SEED
                    +
                    user_id*10000
                    +
                    run_id
                )


                for policy in POLICIES:


                    output = run_policy(
                        theta_star,
                        policy,
                        T,
                        seed
                    )


                    results.append(
                        {

                            "user_id":
                                user_id,

                            "T":
                                T,

                            "run":
                                run_id,

                            "policy":
                                policy,

                            **output

                        }
                    )



    results = pd.DataFrame(
        results
    )


    file = (
        OUTPUT_DIR
        /
        "non_uniform_ablation.csv"
    )


    results.to_csv(
        file,
        index=False
    )


    print(
        "Saved:",
        file
    )


    print(
        results.groupby(
            [
                "T",
                "policy"
            ]
        )
        .mean()
    )



if __name__ == "__main__":

    run()