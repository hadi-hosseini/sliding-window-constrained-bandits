
# Linear Bandits under Exact Sliding-Window Constraints

This repository accompanies the paper **[Linear Bandits under Exact Sliding-Window Constraints](https://arxiv.org/abs/2610.08745)**.

## Overview

We study linear bandit problems under **exact sliding-window constraints**, where every consecutive block of actions must satisfy a prescribed feasibility condition.

Our work investigates offline optimization and online learning under these constraints. We characterize when stationary solutions are optimal or near-optimal in the offline setting and show that geometric structure alone may be insufficient to guarantee sublinear regret in the online setting.

We develop algorithms based on optimistic optimization and controlled policy switching, obtaining regret guarantees that account for both statistical complexity and the difficulty of transitioning between feasible actions. We also extend the framework to general sliding-window constraints using a finite-memory control formulation.


## Citation

If you use this work in your research, please cite the following preprint:

```bibtex
@misc{hosseini2026linearbandits,
  title  = {Linear Bandits under Exact Sliding-Window Constraints},
  author = {Hosseini, Seyed Mohammad Hadi and
            Abbasi-Yadkori, Yasin and
            Vakili, Sattar},
  year   = {2026},
  url    = {https://arxiv.org/abs/2610.08745}
}
```
