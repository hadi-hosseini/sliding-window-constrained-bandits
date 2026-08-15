import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from rare_switching import (
    THETA_STAR,
    W,
    T,
    LAMBDA_REG,
    DELTA,
    optimistic_simplex_action,
    transition_path,
    count_window_violations,
)


RHO_VALUES = [0.25, 0.5, 1.0, 2.0]

ALPHA = 2.0
N_RUNS = 30
SEED = 42
NOISE_STD = 0.1


def run_rho_experiment(
    noise,
    rho,
):
    d = len(THETA_STAR)

    V = LAMBDA_REG * np.eye(d)
    b = np.zeros(d)

    actions = np.zeros(
        (T, d)
    )

    optimal_reward = np.max(
        THETA_STAR
    )

    pseudo_regret = 0.0
    target_switches = 0
    transition_rounds = 0

    t = 0
    episode = 1
    previous_target = None

    while t < T:

        _, episode_logdet = (
            np.linalg.slogdet(V)
        )

        z, _ = optimistic_simplex_action(
            V=V,
            b=b,
            lambda_reg=LAMBDA_REG,
            delta=DELTA,
        )

        if (
            previous_target is not None
            and not np.allclose(
                z,
                previous_target,
                atol=1e-12,
                rtol=0.0,
            )
        ):
            target_switches += 1

            path = transition_path(
                z=previous_target,
                z_next=z,
                w=W,
                rho=rho,
            )

            for action in path:

                if t >= T:
                    break

                actions[t] = action

                reward = (
                    np.dot(
                        THETA_STAR,
                        action,
                    )
                    + noise[t]
                )

                pseudo_regret += (
                    optimal_reward
                    - np.dot(
                        THETA_STAR,
                        action,
                    )
                )

                transition_rounds += 1
                t += 1

        if t >= T:
            break

        stationary_rounds = 0

        while t < T:

            _, current_logdet = (
                np.linalg.slogdet(V)
            )

            determinant_condition = (
                current_logdet
                <= episode_logdet
                + np.log(ALPHA)
                + 1e-12
            )

            warmup_condition = (
                episode == 1
                and stationary_rounds < W - 1
            )

            if not (
                determinant_condition
                or warmup_condition
            ):
                break

            action = z

            actions[t] = action

            reward = (
                np.dot(
                    THETA_STAR,
                    action,
                )
                + noise[t]
            )

            pseudo_regret += (
                optimal_reward
                - np.dot(
                    THETA_STAR,
                    action,
                )
            )

            V += np.outer(
                action,
                action,
            )

            b += (
                reward * action
            )

            stationary_rounds += 1
            t += 1

        previous_target = z.copy()
        episode += 1

    violations = count_window_violations(
        actions=actions,
        w=W,
        rho=rho,
    )

    return {
        "regret": float(
            pseudo_regret
        ),
        "switches": int(
            target_switches
        ),
        "transition_rounds": int(
            transition_rounds
        ),
        "violations": int(
            violations
        ),
    }


def mean_and_ci(values):

    values = np.asarray(
        values
    )

    mean = np.mean(values)

    if len(values) <= 1:
        return mean, 0.0

    se = (
        np.std(
            values,
            ddof=1,
        )
        / np.sqrt(
            len(values)
        )
    )

    return mean, 1.96 * se


def run_sensitivity():

    metrics = [
        "regret",
        "switches",
        "transition_rounds",
        "violations",
    ]

    results = {
        rho: {
            metric: np.zeros(
                N_RUNS
            )
            for metric in metrics
        }
        for rho in RHO_VALUES
    }

    for run in tqdm(
        range(N_RUNS),
        desc="Ramping-radius sensitivity",
    ):

        rng = np.random.default_rng(
            SEED + run
        )

        noise = rng.normal(
            0.0,
            NOISE_STD,
            T,
        )

        for rho in RHO_VALUES:

            result = run_rho_experiment(
                noise=noise,
                rho=rho,
            )

            for metric in metrics:
                results[rho][metric][run] = (
                    result[metric]
                )

    return results


def print_results(results):

    print()
    print(
        "Sensitivity 2: Ramping Radius"
    )
    print()

    for rho in RHO_VALUES:

        regret, regret_ci = mean_and_ci(
            results[rho]["regret"]
        )

        switches, switches_ci = mean_and_ci(
            results[rho]["switches"]
        )

        transitions, transitions_ci = (
            mean_and_ci(
                results[rho][
                    "transition_rounds"
                ]
            )
        )

        violations, violations_ci = (
            mean_and_ci(
                results[rho][
                    "violations"
                ]
            )
        )

        print(
            f"rho={rho:>4.2f} | "
            f"regret={regret:.2f} +/- {regret_ci:.2f} | "
            f"switches={switches:.2f} +/- {switches_ci:.2f} | "
            f"transitions={transitions:.2f} +/- {transitions_ci:.2f} | "
            f"violations={violations:.2f} +/- {violations_ci:.2f}"
        )


def plot_results(results):

    rhos = np.asarray(
        RHO_VALUES
    )

    metrics = [
        (
            "regret",
            "Pseudo-regret",
        ),
        (
            "switches",
            "Target switches",
        ),
        (
            "transition_rounds",
            "Transition rounds",
        ),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10, 3.2),
    )

    for ax, (
        metric,
        ylabel,
    ) in zip(
        axes,
        metrics,
    ):

        means = []
        cis = []

        for rho in RHO_VALUES:

            mean, ci = mean_and_ci(
                results[rho][metric]
            )

            means.append(mean)
            cis.append(ci)

        means = np.asarray(
            means
        )

        cis = np.asarray(
            cis
        )

        ax.plot(
            rhos,
            means,
            marker="o",
        )

        ax.fill_between(
            rhos,
            means - cis,
            means + cis,
            alpha=0.35,
        )

        ax.set_xlabel(
            r"Ramping radius $\rho$"
        )

        ax.set_ylabel(
            ylabel
        )

    plt.tight_layout()

    plt.savefig(
        "imgs/sensitivity2.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":

    results = run_sensitivity()

    print_results(
        results
    )

    plot_results(
        results
    )