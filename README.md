# nopt
Nonlinear optimal control via nonlinear programming (NLP). This library relies upon jax to compute gradients, Jacobian-vector products, and Hessian-vector products through automatic differentiation. Because of this, the solution converges in a small number of iterations, but the runtime suffers.

## Usage

First create an optimal control problem by subclassing from the template given in [core.py](nopt/core.py). See [inverted_pendulum.py](nopt/problems/inverted_pendulum.py) for an example.

``` python
import jax.numpy as np
from nopt.problems import InvertedPendulum

ip = InvertedPendulum()
```

Specifying boundary conditions and number of grid points:

``` python
bcs = {'x0': jnp.array([jnp.pi, 0.]),
       'xN': jnp.array([0., 0.])
N = 10
```

Create a NLP problem:
``` python
from nopt import NlpProblem

problem = NlpProblem(ip, boundary_conditions=bcs, N=N)
```
Then find the optimized solution:

``` python
from nopt import solve

zstar = solve(problem, max_iters=10)
```

You can visualize the optimized output by calling ```plot```:

``` python
anim = problem.plot(zstar)
```

See the [example notebook](example.ipynb).

![](fig/inverted_pendulum.gif)
