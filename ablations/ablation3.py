import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from rare_switching import (
    THETA_STAR,
    W,
    T,
    RHO,
    LAMBDA_REG,
    DELTA,
    optimistic_simplex_action,
    transition_path,
    count_window_violations,
)


ALPHA = 2.0
N_RUNS = 30
SEED = 42
NOISE_STD = 0.1

TRANSITION_MODES = [
    "proposed",
    "linear",
]


def alternative_transition_path(
    z,
    z_next,
    w,
    rho,
):
    distance = np.linalg.norm(
        z_next - z,
        ord=1,
    )

    if distance <= 1e-12:
        return []

    n_steps = int(
        np.ceil(distance / rho)
    )

    n_steps = max(n_steps, 1)

    path = []

    for j in range(1, n_steps + 1):
        fraction = j / n_steps

        action = (
            (1.0 - fraction) * z
            + fraction * z_next
        )

        path.append(action)

    return path


def run_transition_ablation(
    noise,
    transition_mode,
):
    d = len(THETA_STAR)

    V = LAMBDA_REG * np.eye(d)
    b = np.zeros(d)

    actions = np.zeros((T, d))

    optimal_reward = np.max(THETA_STAR)

    pseudo_regret = 0.0
    target_switches = 0
    transition_rounds = 0

    t = 0
    episode = 1
    previous_target = None

    while t < T:

        _, episode_logdet = np.linalg.slogdet(V)

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

            if transition_mode == "proposed":
                path = transition_path(
                    z=previous_target,
                    z_next=z,
                    w=W,
                    rho=RHO,
                )

            else:
                path = alternative_transition_path(
                    z=previous_target,
                    z_next=z,
                    w=W,
                    rho=RHO,
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

            _, current_logdet = np.linalg.slogdet(V)

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

            b += reward * action

            stationary_rounds += 1
            t += 1

        previous_target = z.copy()
        episode += 1

    violations = count_window_violations(
        actions=actions,
        w=W,
        rho=RHO,
    )

    return {
        "regret": float(pseudo_regret),
        "switches": int(target_switches),
        "transition_rounds": int(
            transition_rounds
        ),
        "violations": int(violations),
    }


def mean_and_ci(values):

    values = np.asarray(values)

    mean = np.mean(values)

    if len(values) <= 1:
        return mean, 0.0

    se = (
        np.std(
            values,
            ddof=1,
        )
        / np.sqrt(len(values))
    )

    return mean, 1.96 * se


def run_ablation():

    metrics = [
        "regret",
        "switches",
        "transition_rounds",
        "violations",
    ]

    results = {
        mode: {
            metric: np.zeros(N_RUNS)
            for metric in metrics
        }
        for mode in TRANSITION_MODES
    }

    for run in tqdm(
        range(N_RUNS),
        desc="Transition-mechanism ablation",
    ):

        rng = np.random.default_rng(
            SEED + run
        )

        noise = rng.normal(
            0.0,
            NOISE_STD,
            T,
        )

        for mode in TRANSITION_MODES:

            result = run_transition_ablation(
                noise=noise,
                transition_mode=mode,
            )

            for metric in metrics:
                results[mode][metric][run] = (
                    result[metric]
                )

    return results


def print_results(results):

    print()
    print(
        "Ablation 3: Transition Mechanism"
    )
    print()

    for mode in TRANSITION_MODES:

        regret, regret_ci = mean_and_ci(
            results[mode]["regret"]
        )

        switches, switches_ci = mean_and_ci(
            results[mode]["switches"]
        )

        transitions, transitions_ci = (
            mean_and_ci(
                results[mode][
                    "transition_rounds"
                ]
            )
        )

        violations, violations_ci = (
            mean_and_ci(
                results[mode]["violations"]
            )
        )

        print(
            f"{mode:>12} | "
            f"regret={regret:.2f} +/- {regret_ci:.2f} | "
            f"switches={switches:.2f} +/- {switches_ci:.2f} | "
            f"transitions={transitions:.2f} +/- {transitions_ci:.2f} | "
            f"violations={violations:.2f} +/- {violations_ci:.2f}"
        )


def plot_results(results):

    labels = [
        "Proposed",
        "Alternative",
    ]

    x = np.arange(
        len(TRANSITION_MODES)
    )

    metrics = [
        ("regret", "Pseudo-regret"),
        ("transition_rounds", "Transition rounds"),
        ("violations", "Constraint violations"),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10, 3.2),
    )

    for ax, (metric, ylabel) in zip(
        axes,
        metrics,
    ):

        means = []
        cis = []

        for mode in TRANSITION_MODES:

            mean, ci = mean_and_ci(
                results[mode][metric]
            )

            means.append(mean)
            cis.append(ci)

        means = np.asarray(means)
        cis = np.asarray(cis)

        ax.plot(
            x,
            means,
            marker="o",
        )

        ax.fill_between(
            x,
            means - cis,
            means + cis,
            alpha=0.35,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        ax.set_ylabel(ylabel)

    plt.tight_layout()

    plt.savefig(
        "imgs/ablation3.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":

    results = run_ablation()

    print_results(results)

    plot_results(results)