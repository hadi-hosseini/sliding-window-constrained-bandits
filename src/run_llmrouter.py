from pathlib import Path
import numpy as np
import pandas as pd
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
    "./llmrouter_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


W = 5


BUDGET_VALUES = [
    0.15,
    0.10,
    0.08,
    0.06,
    0.04
]


T_VALUES = [
    50,
    100,
    250,
    500,
    750,
    1000
]


N_RUNS = 10


LAMBDA_REG = 1.0

DELTA = 0.05

NOISE_R = 0.5

DET_FACTOR = 2.0

SEED = 42


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

        self.infeasible_steps = 0



    def feasible_actions(
        self,
        t
    ):

        q = self.query_order[t]


        recent = self.history[
            -(W-1):
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


    def __init__(self):

        self.V = (
            LAMBDA_REG
            *
            np.eye(d)
        )


        self.b = np.zeros(d)



    def theta_hat(self):

        return np.linalg.solve(
            self.V,
            self.b
        )



    def beta(self):

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



    def logdet(self):

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


        self.b += y*x

class MyopicFeasibleOFUL:


    def __init__(self):

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
            np.argmax(scores)
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


    def __init__(self):

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



        # no feasible model
        if len(actions) == 0:

            env.infeasible_steps += 1

            continue



        action = agent.choose(
            actions,
            q
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

        "expected_reward":
        env.expected_reward,


        "average_reward":
        env.expected_reward / T,


        "violations":
        env.violations,


        "infeasible_steps":
        env.infeasible_steps,


        "updates":
        agent.n_updates

    }



def run():

    results = []


    partial_file = (
        OUTPUT_DIR /
        "experiment_results_budget_partial.csv"
    )


    final_file = (
        OUTPUT_DIR /
        "experiment_results_budget.csv"
    )


    for budget in BUDGET_VALUES:


        print(
            "\n=============================="
        )

        print(
            "Running budget:",
            budget
        )

        print(
            "=============================="
        )


        for T in tqdm(
            T_VALUES,
            desc=f"Budget {budget}"
        ):


            for run_id in range(
                N_RUNS
            ):


                for policy in POLICIES:


                    result = run_policy(
                        policy,
                        T,
                        SEED + run_id,
                        budget
                    )


                    results.append(
                        {
                            "budget": budget,
                            "T": T,
                            "run": run_id,
                            "policy": policy,
                            **result
                        }
                    )



        checkpoint = pd.DataFrame(
            results
        )


        checkpoint.to_csv(
            partial_file,
            index=False
        )


        print(
            "Checkpoint saved:",
            partial_file
        )


        print(
            checkpoint
            .groupby(
                [
                    "budget",
                    "T",
                    "policy"
                ]
            )
            .mean()
        )



    results = pd.DataFrame(
        results
    )


    results.to_csv(
        final_file,
        index=False
    )


    print(
        "\nFinal saved:",
        final_file
    )


    print(
        results
        .groupby(
            [
                "budget",
                "T",
                "policy"
            ]
        )
        .mean()
    )


if __name__ == "__main__":
    run()