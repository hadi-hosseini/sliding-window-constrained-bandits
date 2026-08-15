import numpy as np


THETA_STAR = np.array(
    [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00],
    dtype=float,
)

W = 8
T = 16384
RHO = 1.0
LAMBDA_REG = 1.0
DELTA = 0.05


def optimistic_simplex_action(
    V,
    b,
    lambda_reg,
    delta,
):
    d = len(b)

    theta_hat = np.linalg.solve(V, b)

    sign, logdet = np.linalg.slogdet(V)

    beta = np.sqrt(
        lambda_reg
    ) + np.sqrt(
        2.0 * np.log(1.0 / delta)
        + d * np.log(
            1.0 + np.exp(logdet / d) / lambda_reg
        )
    )

    V_inv = np.linalg.inv(V)

    scores = np.zeros(d)

    for i in range(d):
        scores[i] = (
            theta_hat[i]
            + beta * np.sqrt(V_inv[i, i])
        )

    z = np.zeros(d)
    z[np.argmax(scores)] = 1.0

    return z, np.argmax(scores)


def transition_path(
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
        np.ceil(
            (w - 1) * distance / rho
        )
    )

    n_steps = max(n_steps, 1)

    path = []

    for j in range(1, n_steps + 1):
        alpha = j / n_steps

        action = (
            (1.0 - alpha) * z
            + alpha * z_next
        )

        path.append(action)

    return path


def count_window_violations(
    actions,
    w,
    rho,
):
    T_current = len(actions)

    violations = 0

    for t in range(w - 1, T_current):

        window = actions[
            t - w + 1 : t + 1
        ]

        if np.any(
            np.linalg.norm(
                window[1:] - window[:-1],
                axis=1,
                ord=1,
            ) > rho + 1e-10
        ):
            violations += 1

    return violations


def run_rare_switching_oful(
    noise,
    rho=RHO,
    alpha=2.0,
    theta_star=THETA_STAR,
    w=W,
    lambda_reg=LAMBDA_REG,
    delta=DELTA,
):
    T_current = len(noise)
    d = len(theta_star)

    V = lambda_reg * np.eye(d)
    b = np.zeros(d)

    actions = np.zeros(
        (T_current, d)
    )

    optimal_reward = np.max(theta_star)

    pseudo_regret = 0.0
    target_switches = 0
    transition_rounds = 0

    t = 0
    episode = 1
    previous_target = None

    while t < T_current:

        _, episode_logdet = np.linalg.slogdet(V)

        z, _ = optimistic_simplex_action(
            V=V,
            b=b,
            lambda_reg=lambda_reg,
            delta=delta,
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
                w=w,
                rho=rho,
            )

            for action in path:

                if t >= T_current:
                    break

                actions[t] = action

                reward = (
                    np.dot(
                        theta_star,
                        action,
                    )
                    + noise[t]
                )

                pseudo_regret += (
                    optimal_reward
                    - np.dot(
                        theta_star,
                        action,
                    )
                )

                V += np.outer(
                    action,
                    action,
                )

                b += reward * action

                transition_rounds += 1
                t += 1

        if t >= T_current:
            break

        stationary_rounds = 0

        while t < T_current:

            _, current_logdet = (
                np.linalg.slogdet(V)
            )

            determinant_condition = (
                current_logdet
                <= episode_logdet
                + np.log(alpha)
                + 1e-12
            )

            warmup_condition = (
                episode == 1
                and stationary_rounds < w - 1
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
                    theta_star,
                    action,
                )
                + noise[t]
            )

            pseudo_regret += (
                optimal_reward
                - np.dot(
                    theta_star,
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
        w=w,
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