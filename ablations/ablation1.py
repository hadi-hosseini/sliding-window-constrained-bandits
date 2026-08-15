import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from rare_switching import (
    THETA_STAR,
    W,
    T,
    RHO,
    run_rare_switching_oful,
)


ALPHA_VALUES = [
    1.25,
    1.5,
    2.0,
    4.0,
    8.0,
]

N_RUNS = 30
SEED = 42


def mean_and_ci(values):
    values = np.asarray(values)

    mean = np.mean(values)

    if len(values) <= 1:
        return mean, 0.0

    se = (
        np.std(values, ddof=1)
        / np.sqrt(len(values))
    )

    return mean, 1.96 * se


def run_switching_threshold_ablation():
    metrics = [
        "regret",
        "switches",
        "transition_rounds",
        "violations",
    ]

    results = {
        alpha: {
            metric: np.zeros(N_RUNS)
            for metric in metrics
        }
        for alpha in ALPHA_VALUES
    }

    for run in tqdm(
        range(N_RUNS),
        desc="Switching-threshold ablation",
    ):
        rng = np.random.default_rng(
            SEED + run
        )

        noise = rng.normal(
            0.0,
            0.1,
            T,
        )

        for alpha in ALPHA_VALUES:

            result = run_rare_switching_oful(
                noise=noise,
                rho=RHO,
                alpha=alpha,
                theta_star=THETA_STAR,
                w=W,
            )

            for metric in metrics:
                results[alpha][metric][run] = (
                    result[metric]
                )

    return results


def print_results(results):

    print()
    print(
        "Switching-threshold ablation"
    )
    print()

    for alpha in ALPHA_VALUES:

        regret, regret_ci = mean_and_ci(
            results[alpha]["regret"]
        )

        switches, switches_ci = mean_and_ci(
            results[alpha]["switches"]
        )

        transitions, transitions_ci = (
            mean_and_ci(
                results[alpha][
                    "transition_rounds"
                ]
            )
        )

        violations, violations_ci = (
            mean_and_ci(
                results[alpha][
                    "violations"
                ]
            )
        )

        print(
            f"alpha={alpha:.2f} | "
            f"regret={regret:.2f} +/- {regret_ci:.2f} | "
            f"switches={switches:.2f} +/- {switches_ci:.2f} | "
            f"transitions={transitions:.2f} +/- {transitions_ci:.2f} | "
            f"violations={violations:.2f} +/- {violations_ci:.2f}"
        )


def plot_results(results):

    alphas = np.array(
        ALPHA_VALUES
    )

    regret_means = []
    regret_cis = []

    switch_means = []
    switch_cis = []

    transition_means = []
    transition_cis = []

    for alpha in ALPHA_VALUES:

        mean, ci = mean_and_ci(
            results[alpha]["regret"]
        )

        regret_means.append(mean)
        regret_cis.append(ci)

        mean, ci = mean_and_ci(
            results[alpha]["switches"]
        )

        switch_means.append(mean)
        switch_cis.append(ci)

        mean, ci = mean_and_ci(
            results[alpha][
                "transition_rounds"
            ]
        )

        transition_means.append(mean)
        transition_cis.append(ci)

    regret_means = np.asarray(regret_means)
    regret_cis = np.asarray(regret_cis)

    switch_means = np.asarray(switch_means)
    switch_cis = np.asarray(switch_cis)

    transition_means = np.asarray(
        transition_means
    )
    transition_cis = np.asarray(
        transition_cis
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 3.5),
    )

    axes[0].plot(
        alphas,
        regret_means,
        marker="o",
    )

    axes[0].fill_between(
        alphas,
        regret_means - regret_cis,
        regret_means + regret_cis,
        alpha=0.35,
    )

    axes[0].set_xlabel(
        r"Switching threshold $\alpha$"
    )
    axes[0].set_ylabel(
        "Pseudo-regret"
    )

    axes[1].plot(
        alphas,
        switch_means,
        marker="o",
    )

    axes[1].fill_between(
        alphas,
        switch_means - switch_cis,
        switch_means + switch_cis,
        alpha=0.35,
    )

    axes[1].set_xlabel(
        r"Switching threshold $\alpha$"
    )
    axes[1].set_ylabel(
        "Target switches"
    )

    axes[2].plot(
        alphas,
        transition_means,
        marker="o",
    )

    axes[2].fill_between(
        alphas,
        transition_means - transition_cis,
        transition_means + transition_cis,
        alpha=0.2,
    )

    axes[2].set_xlabel(
        r"Switching threshold $\alpha$"
    )
    axes[2].set_ylabel(
        "Transition rounds"
    )

    plt.tight_layout()

    plt.savefig(
        "imgs/ablation1.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":

    results = (
        run_switching_threshold_ablation()
    )

    print_results(results)

    plot_results(results)