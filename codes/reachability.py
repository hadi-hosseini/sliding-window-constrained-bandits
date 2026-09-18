import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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


THETA_STAR = np.array([0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00 ])
D = 8
W = 8
T = 16384
RHO_VALUES = [0.25, 0.50, 1.00, 2.00]
NOISE_STD = 0.1
LAMBDA_REG = 1.0
DELTA = 0.05
S_BOUND = np.linalg.norm(THETA_STAR)
N_RUNS = 30
BASE_SEED = 2026

POLICY_NAMES = ["Rare-switching OFUL", "Frequent-update feasible OFUL", "Unconstrained OFUL"]

POLICY_COLORS = {
    "Rare-switching OFUL": COLORS["blue"],
    "Frequent-update feasible OFUL": COLORS["orange"],
    "Unconstrained OFUL": COLORS["red"],
}

POLICY_MARKERS = {
    "Rare-switching OFUL": "o",
    "Frequent-update feasible OFUL": "s",
    "Unconstrained OFUL": "^",
}


def confidence_radius(V, lambda_reg=LAMBDA_REG, noise_scale=NOISE_STD, theta_bound=S_BOUND, delta=DELTA):
    d = V.shape[0]
    sign, logdet_v = np.linalg.slogdet(V)

    if sign <= 0:
        raise ValueError("V must be positive definite")

    logdet_initial = d * np.log(lambda_reg)

    log_term = 0.5 * (logdet_v - logdet_initial) + np.log(1.0 / delta)

    beta = (
        noise_scale * np.sqrt(max(0.0, 2.0 * log_term))
        + np.sqrt(lambda_reg) * theta_bound
    )

    return beta

def optimistic_simplex_action(V,b,lambda_reg=LAMBDA_REG,noise_scale=NOISE_STD,theta_bound=S_BOUND,delta=DELTA):
    d=V.shape[0]
    theta_hat=np.linalg.solve(V,b)

    beta=confidence_radius(V=V,lambda_reg=lambda_reg,noise_scale=noise_scale,theta_bound=theta_bound,delta=delta)
    V_inv=np.linalg.inv(V)
    optimistic_values=theta_hat+beta*np.sqrt(np.diag(V_inv))
    index=int(np.argmax(optimistic_values))
    z=np.zeros(d)
    z[index]=1.0
    return z,index


def transition_path(z,z_next,w,rho):
    if np.allclose(z,z_next,atol=1e-12,rtol=0.0):
        return []
    distance=np.sum(np.abs(z_next-z))
    N=int(np.ceil(((w-1)*distance/rho)-1e-12))
    N=max(N,1)
    path=[]
    for j in range(1,N+1):
        y_j=z+(j/N)*(z_next-z)
        path.append(y_j)
    for _ in range(w-2):
        path.append(z_next.copy())
    return path


def count_window_violations(actions,w,rho,tolerance=1e-8):
    T_current=len(actions)

    if T_current<w:
        return 0
    
    adjacent_distances=np.sum(np.abs(actions[1:]-actions[:-1]),axis=1)
    adjacent_windows=np.lib.stride_tricks.sliding_window_view(adjacent_distances,w-1)
    maximum_adjacent=np.max(adjacent_windows,axis=1)
    cyclic_distances=np.sum(np.abs(actions[w-1:]-actions[:T_current-w+1]),axis=1)
    maximum_window_distance=np.maximum(maximum_adjacent,cyclic_distances)
    violations=np.sum(maximum_window_distance>rho+tolerance)

    return int(violations)


def run_rare_switching_oful(noise, rho, theta_star=THETA_STAR, w=W, lambda_reg=LAMBDA_REG, delta=DELTA):
    T_current = len(noise)
    d = len(theta_star)
    V = lambda_reg * np.eye(d)
    b = np.zeros(d)
    actions = np.zeros((T_current, d))
    optimal_reward = np.max(theta_star)
    pseudo_regret = 0.0
    target_switches = 0
    transition_rounds = 0
    t = 0
    episode = 1
    previous_target = None


    while t < T_current:
        _, episode_logdet = np.linalg.slogdet(V)

        z, index = optimistic_simplex_action(V=V, b=b, lambda_reg=lambda_reg, delta=delta)

        if previous_target is not None and not np.allclose(z, previous_target, atol=1e-12, rtol=0.0):
            target_switches += 1
            path = transition_path(z=previous_target, z_next=z, w=w, rho=rho)

            for action in path:
                if t >= T_current:
                    break

                actions[t] = action
                reward = np.dot(theta_star, action) + noise[t]
                pseudo_regret += optimal_reward - np.dot(theta_star, action)
                transition_rounds += 1
                t += 1

        if t >= T_current:
            break

        stationary_rounds = 0

        while t < T_current:
            _, current_logdet = np.linalg.slogdet(V)

            determinant_condition = current_logdet <= episode_logdet + np.log(2.0) + 1e-12
            warmup_condition = episode == 1 and stationary_rounds < w - 1
            if not (determinant_condition or warmup_condition):
                break

            action = z
            actions[t] = action
            reward = np.dot(theta_star, action) + noise[t]
            pseudo_regret += optimal_reward - np.dot(theta_star, action)
            V += np.outer(action, action)
            b += reward * action
            stationary_rounds += 1
            t += 1

        previous_target = z.copy()
        episode += 1

    violations = count_window_violations(actions=actions, w=w, rho=rho)

    return {
        "regret": float(pseudo_regret),
        "switches": int(target_switches),
        "transition_rounds": int(transition_rounds),
        "violations": int(violations),
    }


def run_frequent_update_feasible_oful(noise, rho, theta_star=THETA_STAR, w=W, lambda_reg=LAMBDA_REG, delta=DELTA):
    T_current = len(noise)
    d = len(theta_star)
    V = lambda_reg * np.eye(d)
    b = np.zeros(d)
    actions = np.zeros((T_current, d))
    optimal_reward = np.max(theta_star)
    pseudo_regret = 0.0
    target_switches = 0
    transition_rounds = 0
    t = 0

    z, _ = optimistic_simplex_action(V=V, b=b, lambda_reg=lambda_reg, delta=delta)
    initial_rounds = min(w - 1, T_current)

    for _ in range(initial_rounds):
        action = z
        actions[t] = action
        reward = np.dot(theta_star, action) + noise[t]
        pseudo_regret += optimal_reward - np.dot(theta_star, action)
        V += np.outer(action, action)
        b += reward * action
        t += 1
    previous_target = z.copy()

    while t < T_current:
        z, _ = optimistic_simplex_action(V=V, b=b, lambda_reg=lambda_reg, delta=delta)

        if not np.allclose(z, previous_target, atol=1e-12, rtol=0.0):
            target_switches += 1

            path = transition_path(z=previous_target, z_next=z, w=w, rho=rho)

            for action in path:
                if t >= T_current:
                    break

                actions[t] = action
                reward = np.dot(theta_star, action) + noise[t]
                pseudo_regret += optimal_reward - np.dot(theta_star, action)
                transition_rounds += 1
                t += 1

        if t >= T_current:
            break

        action = z
        actions[t] = action
        reward = np.dot(theta_star, action) + noise[t]
        pseudo_regret += optimal_reward - np.dot(theta_star, action)
        V += np.outer(action, action)
        b += reward * action
        t += 1
        previous_target = z.copy()

    violations = count_window_violations(actions=actions, w=w, rho=rho)

    return {
        "regret": float(pseudo_regret),
        "switches": int(target_switches),
        "transition_rounds": int(transition_rounds),
        "violations": int(violations),
    }


def run_unconstrained_oful(noise, rho, theta_star=THETA_STAR, w=W, lambda_reg=LAMBDA_REG, delta=DELTA):
    T_current = len(noise)
    d = len(theta_star)
    V = lambda_reg * np.eye(d)
    b = np.zeros(d)
    actions = np.zeros((T_current, d))
    optimal_reward = np.max(theta_star)
    pseudo_regret = 0.0
    target_switches = 0
    previous_index = None

    for t in range(T_current):
        z, index = optimistic_simplex_action(V=V, b=b, lambda_reg=lambda_reg, delta=delta)

        if previous_index is not None and index != previous_index:
            target_switches += 1

        actions[t] = z
        reward = np.dot(theta_star, z) + noise[t]
        pseudo_regret += optimal_reward - np.dot(theta_star, z)
        V += np.outer(z, z)
        b += reward * z
        previous_index = index

    violations = count_window_violations(actions=actions, w=w, rho=rho)

    return {
        "regret": float(pseudo_regret),
        "switches": int(target_switches),
        "transition_rounds": 0,
        "violations": int(violations),
    }


def run_rare_switching_experiment(n_runs=N_RUNS, seed=BASE_SEED):
    metrics = ["regret", "switches", "transition_rounds", "violations"]
    results = {policy: {metric: np.zeros((len(RHO_VALUES), n_runs)) for metric in metrics} for policy in POLICY_NAMES}

    for run in tqdm(range(n_runs), desc="Runs"):
        rng = np.random.default_rng(seed + run)
        noise = rng.normal(loc=0.0, scale=NOISE_STD, size=T)

        print(f"Run {run + 1:2d}/{n_runs}")

        for rho_index, rho in enumerate(RHO_VALUES):
            rare_result = run_rare_switching_oful(noise=noise, rho=rho)
            frequent_result = run_frequent_update_feasible_oful(noise=noise, rho=rho)
            unconstrained_result = run_unconstrained_oful(noise=noise, rho=rho)

            policy_results = {
                "Rare-switching OFUL": rare_result,
                "Frequent-update feasible OFUL": frequent_result,
                "Unconstrained OFUL": unconstrained_result,
            }

            for policy in POLICY_NAMES:
                for metric in metrics:
                    results[policy][metric][rho_index, run] = policy_results[policy][metric]

    return results


def mean_and_ci(values):
    mean = np.mean(values, axis=1)

    if values.shape[1] == 1:
        ci = np.zeros_like(mean)

    else:
        standard_error = (np.std(values, axis=1, ddof=1) / np.sqrt(values.shape[1]))
        ci = 1.96 * standard_error

    return mean, ci


def print_experiment_summary(results):
    metrics = ["regret", "switches", "transition_rounds", "violations"]

    for rho_index, rho in enumerate(RHO_VALUES):
        print(f"\nrho = {rho}")
        print("-" * 90)

        for policy in POLICY_NAMES:
            values = []

            for metric in metrics:
                data = (results[policy][metric][rho_index])
                mean = np.mean(data)

                if len(data) > 1:
                    ci = (1.96 * np.std(data, ddof=1) / np.sqrt(len(data)))

                else:
                    ci = 0.0

                values.append((mean, ci))

            print(
                f"{policy:32s} | "
                f"regret "
                f"{values[0][0]:8.2f} "
                f"± {values[0][1]:6.2f} | "
                f"switches "
                f"{values[1][0]:7.2f} "
                f"± {values[1][1]:5.2f} | "
                f"transition "
                f"{values[2][0]:8.2f} "
                f"± {values[2][1]:6.2f} | "
                f"violations "
                f"{values[3][0]:8.2f} "
                f"± {values[3][1]:6.2f}"
            )


def plot_rare_switching_experiment(results):

    metric_info = [
        (
            "regret",
            "(a) Realized Pseudo-Regret",
            "Realized pseudo-regret"
        ),
        (
            "switches",
            "(b) Stationary-Target Switches",
            "Number of target switches"
        ),
        (
            "transition_rounds",
            "(c) Feasible Transition Cost",
            "Transition rounds"
        ),
        (
            "violations",
            "(d) Sliding-Window Violations",
            "Number of violations"
        ),
    ]

    x = np.arange(
        len(RHO_VALUES)
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 9)
    )

    axes = axes.flatten()

    for ax, (
        metric,
        title,
        ylabel
    ) in zip(
        axes,
        metric_info
    ):

        for policy in POLICY_NAMES:

            mean, ci = mean_and_ci(
                results[
                    policy
                ][
                    metric
                ]
            )

            ax.plot(
                x,
                mean,
                color=POLICY_COLORS[policy],
                marker=POLICY_MARKERS[policy],
                markersize=7,
                linewidth=2.1,
                markeredgecolor="white",
                markeredgewidth=0.7,
                label=policy
            )

            ax.fill_between(
                x,
                mean - ci,
                mean + ci,
                color=POLICY_COLORS[policy],
                alpha=0.35
            )


        ax.set_title(
            title,
            fontsize=13,
            pad=10
        )

        ax.set_xlabel(
            "Ramping radius ρ",
            fontsize=12
        )

        ax.set_ylabel(
            ylabel,
            fontsize=12
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            [
                "0.25",
                "0.5",
                "1",
                "2"
            ]
        )

        ax.tick_params(
            axis="both",
            labelsize=10
        )

        ax.grid(
            linestyle="--",
            linewidth=0.7,
            alpha=0.22
        )

        ax.spines[
            "top"
        ].set_visible(
            False
        )

        ax.spines[
            "right"
        ].set_visible(
            False
        )

        ax.set_ylim(
            bottom=0
        )

    handles, labels = (
        axes[0]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(
            0.5,
            1.01
        )
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95
        ],
        h_pad=2.5,
        w_pad=2.5
    )

    fig.savefig(
        "imgs/reachability.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":
    results = run_rare_switching_experiment(n_runs=30, seed=2026)
    print_experiment_summary(results)
    plot_rare_switching_experiment(results)