# PyTorch iCEM Implementation
This repository implements the improved Cross Entropy Method (iCEM) 
with approximate dynamics in pytorch, from [this paper](https://martius-lab.github.io/iCEM/). 

MPPI typically requires actual
trajectory samples, but [this paper](https://martius-lab.github.io/iCEM/)
showed that it could be done with approximate dynamics (such as with a neural network)
using importance sampling.

Thus it can be used in place of other trajectory optimization methods
such as the Cross Entropy Method (CEM), or random shooting.


# Related projects
- [pytorch CEM](https://github.com/LemonPi/pytorch_cem) - alternative sampling based MPC
- [pytorch MPPI](https://github.com/UM-ARM-Lab/pytorch_mppi) - alternative sampling based MPC 
- [iCEM](https://github.com/martius-lab/iCEM) - original paper's numpy implementation and experiments code


# Installation
```shell
pip install pytorch-icem
```
for running tests, install with
```shell
pip install pytorch-icem[test]
```
for development, clone the repository then install in editable mode
```shell
pip install -e .
```

# Usage
See `tests/pendulum_approximate_continuous.py` for usage with a neural network approximating
the pendulum dynamics. Basic use case is shown below

```python
from pytorch_icem import iCEM

# create controller with chosen parameters
ctrl = icem.iCEM(dynamics, terminal_cost, nx, nu, sigma=sigma,
                 warmup_iters=10, online_iters=10,
                 num_samples=N_SAMPLES, num_elites=10, horizon=TIMESTEPS, device=d, )

# assuming you have a gym-like env
obs = env.reset()
for i in range(100):
    action = ctrl.command(obs)
    obs, reward, done, _, _ = env.step(action.cpu().numpy())
```

# Requirements
- pytorch (>= 1.0)
- `next state <- dynamics(state, action)` function (doesn't have to be true dynamics)
    - `state` is `K x nx`, `action` is `K x nu`
- `trajectory cost <- cost(state, action)` function for the whole state action trajectory, T is the horizon
    - `cost` is `K x 1`, state is `K x T x nx`, `action` is `K x T x nu`

# Features
- Parallel/batch pytorch implementation for accelerated sampling
