from pathlib import Path
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm


DATA_DIR = Path("/home/hadi/CRLT/llmrouting_processed")
OUTPUT_DIR = Path("./llmrouter_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CSV_FILE = DATA_DIR / "llmrouterbench.csv"


W = 20
BUDGET = 0.05

T_VALUES = [
    500,
    1000,
    1500,
    2000,
    2500,
    3000,
    4000,
    5000
]

N_RUNS = 10

LAMBDA_REG = 1.0
DELTA = 0.05
NOISE_R = 0.5

RANDOM_SEED = 42

policies = [
    "Rare-Update Optimistic Planning",
    "Myopic Feasible OFUL",
    "Unconstrained OFUL"
]


df = pd.read_csv(CSV_FILE)

df["embedding"] = df["embedding"].apply(
    ast.literal_eval
)


models = sorted(
    df["model"].unique()
)

K = len(models)


model_to_id = {
    m:i for i,m in enumerate(models)
}


d = 384 + K



def make_feature(row):

    x = np.asarray(
        row["embedding"],
        dtype=float
    )

    e = np.zeros(K)

    e[
        model_to_id[row["model"]]
    ] = 1.0

    return np.concatenate(
        [
            x,
            e
        ]
    )



features = {}

for idx,row in tqdm(
    df.iterrows(),
    total=len(df)
):

    features[
        (row.query_id,row.model)
    ] = make_feature(row)



query_ids = sorted(
    df.query_id.unique()
)


reward_table = {}

cost_table = {}


for _,row in df.iterrows():

    key = (
        row.query_id,
        row.model
    )

    reward_table[key] = float(
        row.reward
    )

    cost_table[key] = float(
        row.cost
    )



class LLMRoutingEnvironment:


    def __init__(
        self,
        query_order,
        seed=0
    ):

        self.query_order=query_order

        self.rng=np.random.default_rng(seed)

        self.history=[]

        self.total_reward=0

        self.expected_reward=0

        self.violations=0



    def feasible_actions(self):

        recent = self.history[-(W-1):]


        used_cost=sum(
            x[1]
            for x in recent
        )


        q=self.query_order[
            len(self.history)
        ]


        actions=[]


        for j,m in enumerate(models):

            c = cost_table[
                (q,m)
            ]


            if used_cost+c <= BUDGET:
                actions.append(j)


        return actions



    def step(self,action):

        q=self.query_order[
            len(self.history)
        ]

        model=models[action]


        reward=reward_table[
            (q,model)
        ]

        cost=cost_table[
            (q,model)
        ]


        self.history.append(
            (
                action,
                cost
            )
        )


        self.total_reward += reward

        self.expected_reward += reward


        if len(self.history)>=W:

            window_cost=sum(
                c
                for _,c in self.history[-W:]
            )

            if window_cost>BUDGET:
                self.violations+=1


        return reward



class OptimisticLinearBandit:


    def __init__(self):

        self.V = (
            LAMBDA_REG
            *
            np.eye(d)
        )

        self.b=np.zeros(d)



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
                d*np.log(
                    1/DELTA
                )
            )
        )


    def score(self,x):

        theta=self.theta_hat()

        bonus=self.beta()*np.sqrt(
            x @ np.linalg.solve(
                self.V,
                x
            )
        )

        return theta@x + bonus



    def update(self,x,y):

        self.V += np.outer(
            x,
            x
        )

        self.b += y*x



class OFULPolicy:


    def __init__(self):

        self.model=OptimisticLinearBandit()

        self.n_updates=0



    def choose(self,actions,q):

        values=[]

        for a in actions:

            phi=features[
                (
                    q,
                    models[a]
                )
            ]

            values.append(
                self.model.score(phi)
            )

        self.n_updates+=1

        return actions[
            np.argmax(values)
        ]



    def update(self,x,y):

        self.model.update(x,y)



class RareUpdateOptimisticPlanning(OFULPolicy):


    def __init__(self):

        super().__init__()

        self.frozen_scores=None

        self.last_logdet=None



    def choose(self,actions,q):

        logdet=np.linalg.slogdet(
            self.model.V
        )[1]


        if (
            self.last_logdet is None
            or
            logdet >
            self.last_logdet+np.log(2)
        ):

            self.frozen_scores={}

            for a in actions:

                phi=features[
                    (
                        q,
                        models[a]
                    )
                ]

                self.frozen_scores[a]=(
                    self.model.score(phi)
                )

            self.last_logdet=logdet

            self.n_updates+=1


        return max(
            actions,
            key=lambda a:
            self.frozen_scores[a]
        )



def run_policy(
    policy_name,
    T,
    seed
):

    rng=np.random.default_rng(seed)

    query_order=rng.choice(
        query_ids,
        size=T,
        replace=False
    )


    env=LLMRoutingEnvironment(
        query_order,
        seed
    )


    if policy_name=="Rare-Update Optimistic Planning":

        agent=RareUpdateOptimisticPlanning()

    else:

        agent=OFULPolicy()



    for t in range(T):

        q=query_order[t]


        if policy_name=="Unconstrained OFUL":

            actions=list(range(K))

        else:

            actions=env.feasible_actions()



        action=agent.choose(
            actions,
            q
        )


        reward=env.step(
            action
        )


        phi=features[
            (
                q,
                models[action]
            )
        ]


        agent.update(
            phi,
            reward
        )


    return {
        "reward":
        env.total_reward,

        "average_reward":
        env.total_reward/T,

        "violations":
        env.violations,

        "updates":
        agent.n_updates
    }



def run():

    results=[]

    for T in T_VALUES:
        for run_idx in range(N_RUNS):

            for policy in policies:

                seed=(
                    RANDOM_SEED
                    +
                    run_idx
                )

                out=run_policy(
                    policy,
                    T,
                    seed
                )

                results.append(
                    {
                        "T":T,
                        "run":run_idx,
                        "policy":policy,
                        **out
                    }
                )


    results=pd.DataFrame(results)

    results.to_csv(
        OUTPUT_DIR /
        "experiment_results.csv",
        index=False
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


if __name__=="__main__":
    run()