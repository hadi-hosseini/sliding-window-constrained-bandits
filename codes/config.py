from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KuaiRandConfig:
    # Paths
    data_dir: Path = Path("./kuairand_processed")
    output_dir: Path = Path("./kuairand_results")

    # Policies
    policies: tuple[str, ...] = ("Rare-Update Optimistic Planning", "Myopic Feasible OFUL", "Primal-Dual OFUL", "Unconstrained OFUL")
    # policies: tuple[str, ...] = ("Rare-Update Optimistic Planning", "Myopic Feasible OFUL", "Unconstrained OFUL")
    # policies: tuple[str, ...] = ("Rare-Update Optimistic Planning",)


    # Sliding-window constraint
    window_size: int = 20
    max_exposure: int = 5

    # Experiment Parameters
    t_values: tuple[int, ...] = (500, 2500, 5000)
    n_users: int = 100 # 100
    n_runs: int = 10
    min_category_interactions: int = 50
    random_seed: int = 42
    planning_horizon: int = 5

    # Dataset
    theta_column: str = "theta_hat"

    # OFUL
    lambda_reg: float = 1.0
    delta: float = 0.05
    noise_r: float = 0.5
    det_factor: float = 2.0

    # Primal-dual baseline
    dual_eta: float = 0.01


@dataclass(frozen=True)
class LLMRouterConfig:
    # Paths
    data_dir: Path = Path("./llmrouting_processed")
    output_dir: Path = Path("./llmrouter_results")

    # Dataset
    csv_file: str = "llmrouterbench_final.csv"
    feature_file: str = "features.npy"

    # Policies
    policies: tuple[str, ...] = ("Myopic Rare-Update Optimistic Planning", "Myopic Feasible OFUL", "Primal-Dual OFUL", "Unconstrained OFUL")
    # policies: tuple[str, ...] = ("Myopic Feasible OFUL", "Primal-Dual OFUL", "Unconstrained OFUL",)
    # policies: tuple[str, ...] = ("Myopic Rare-Update Optimistic Planning",)


    # Sliding-window budget constraint
    window_size: int = 5
    budget_values: tuple[float, ...] = (0.15, 0.25, 0.35, 0.40,)
    # budget_values: tuple[float, ...] = (0.40,)
    

    # Experiment parameters
    t_values: tuple[int, ...] = (50, 100, 250, 500, 750, 1000, 2500, 5000)
    # t_values: tuple[int, ...] = (5000,)
    n_runs: int = 10
    random_seed: int = 42

    # Rare-update optimistic planning
    det_factor: float = 2.0

    # OFUL
    lambda_reg: float = 1.0
    delta: float = 0.05
    noise_r: float = 0.5
    planning_horizon: int = 1

    # Primal-dual baseline
    dual_eta: float = 5