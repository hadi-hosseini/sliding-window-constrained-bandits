import numpy as np

THETA_STAR = np.array([0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00], dtype=float)
W = 8
T = 16384
RHO = 1.0
NOISE_STD = 0.1
LAMBDA_REG = 1.0
DELTA = 0.05
S_BOUND = np.linalg.norm(THETA_STAR)


def confidence_radius(V, lambda_reg=LAMBDA_REG, noise_scale=NOISE_STD, theta_bound=S_BOUND, delta=DELTA):
    d = V.shape[0]
    sign, logdet_v = np.linalg.slogdet(V)

    if sign <= 0:
        raise ValueError("V must be positive definite")

    logdet_initial = d * np.log(lambda_reg)
    log_term = 0.5 * (logdet_v - logdet_initial) + np.log(1.0 / delta)
    beta = (noise_scale * np.sqrt(max(0.0, 2.0 * log_term)) + np.sqrt(lambda_reg) * theta_bound)

    return beta


def optimistic_simplex_action(V, b, lambda_reg=LAMBDA_REG, noise_scale=NOISE_STD, theta_bound=S_BOUND, delta=DELTA):
    d = len(b)
    theta_hat = np.linalg.solve(V, b)
    beta = confidence_radius(V=V, lambda_reg=lambda_reg, noise_scale=noise_scale, theta_bound=theta_bound, delta=delta)
    V_inv = np.linalg.solve(V, np.eye(d))
    optimistic_values = theta_hat + beta * np.sqrt(np.diag(V_inv))
    index = int(np.argmax(optimistic_values))
    z = np.zeros(d)
    z[index] = 1.0

    return z, index


def transition_path(z, z_next, w, rho):
    distance = np.linalg.norm(z_next - z, ord=1)

    if distance <= 1e-12:
        return []

    n_steps = int(np.ceil((w - 1) * distance / rho))
    n_steps = max(n_steps, 1)
    path = []

    for j in range(1, n_steps + 1):
        alpha = j / n_steps
        action = (1.0 - alpha) * z + alpha * z_next
        path.append(action.copy())

    for _ in range(w - 2):
        path.append(z_next.copy())

    return path


def count_window_violations(actions, w, rho, tolerance=1e-10):
    T_current = len(actions)

    if T_current < w:
        return 0

    violations = 0
    for t in range(w - 1, T_current):
        window = actions[t - w + 1:t + 1]
        adjacent_distances = np.linalg.norm(window[1:] - window[:-1], ord=1, axis=1,)
        cyclic_distance = np.linalg.norm(window[-1] - window[0], ord=1)
        if (np.max(adjacent_distances) > rho + tolerance or cyclic_distance > rho + tolerance):
            violations += 1

    return violations


def run_rare_switching_oful(noise, rho=RHO, alpha=2.0, theta_star=THETA_STAR, w=W, lambda_reg=LAMBDA_REG, noise_scale=NOISE_STD, theta_bound=None, delta=DELTA):
    T_current = len(noise)
    d = len(theta_star)

    if theta_bound is None:
        theta_bound = np.linalg.norm(theta_star)

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
        sign, episode_logdet = np.linalg.slogdet(V)

        if sign <= 0:
            raise ValueError("V must be positive definite")

        z, _ = optimistic_simplex_action(V=V, b=b, lambda_reg=lambda_reg, noise_scale=noise_scale, theta_bound=theta_bound, delta=delta)

        if (previous_target is not None and not np.allclose(z, previous_target, atol=1e-12, rtol=0.0)):
            target_switches += 1
            path = transition_path(z=previous_target, z_next=z, w=w, rho=rho,)

            for action in path:
                if t >= T_current:
                    break

                actions[t] = action
                pseudo_regret += (optimal_reward - np.dot(theta_star, action))
                transition_rounds += 1
                t += 1

        if t >= T_current:
            break

        stationary_rounds = 0

        while t < T_current:
            sign, current_logdet = np.linalg.slogdet(V)

            if sign <= 0:
                raise ValueError("V must be positive definite")

            determinant_condition = (current_logdet <= episode_logdet + np.log(alpha) + 1e-12)
            warmup_condition = (episode == 1 and stationary_rounds < w - 1)

            if not (determinant_condition or warmup_condition):
                break

            action = z
            actions[t] = action
            reward = np.dot(theta_star, action) + noise[t]
            pseudo_regret += (optimal_reward - np.dot(theta_star, action))
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