from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax.lax import stop_gradient
from jax.scipy.sparse.linalg import cg, gmres
from jax import grad, jit, jvp, vjp, random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from nopt.linopt import LinearOperator


class OptControlProblem(ABC):
    """
    Base optimal control problem
    """

    @abstractmethod
    def f(self, x, u):
        """
        Differential equation describing dynamical system
        """
        pass

    @abstractmethod
    def L(self, x, u):
        """
        L in cost function:
          J = phi(x_N) + int_{0}^T L(x(t), u(t)) dt
        """
        pass

    @abstractmethod
    def phi(self, xN):
        """
        phi in cost function:
          J = phi(x_N) + int_{0}^T L(x(t), u(t)) dt
        """
        pass

    @abstractmethod
    def render(self, x, fig, artist):
        pass

    @property
    @abstractmethod
    def statedim(self):
        """
        Dimension of state vector
        """
        pass

    @property
    @abstractmethod
    def inputdim(self):
        """
        Dimension of input control vector
        """
        pass


class NlpProblem:
    """
    Transcribes an optimal control problem into a nonlinear programming (NLP) problem
    """
    def __init__(self, ocp: OptControlProblem, boundary_conditions: dict, N: int=20, collocation_fn=None):

        self.ocp = ocp
        self.bc = {key: jnp.asarray(bc) for key, bc in boundary_conditions.items()}
        self.N = N
        self.dt = 1./N

        if collocation_fn is None:
            def collocation_fn(x, u, h=self.dt):
                return h * self.ocp.f(x, u)
            
        self.collocation_fn = collocation_fn
        

    def _splitz(self, z):
        assert(len(z) == self.nvars)
        xs = z[:(self.N+1)*self.ocp.statedim]
        us = z[-(self.N*self.ocp.inputdim):]
        return xs, us

    @property
    def statedim(self):
        return self.ocp.statedim

    @property
    def inputdim(self):
        return self.ocp.inputdim
    
    @property
    def nvars(self):
        try:
            return self._n
        except:
            self._n = (self.N+1)*self.statedim + self.N*self.inputdim
            return self._n

    @property
    def nconstraints(self):
        try:
            return self._m
        except:
            self._m = len(self.c(jnp.zeros(self.nvars)))
            return self._m
    
    def F(self, z):
        try:
            return self._F(z)
        
        except:
            
            def _F(z):
                xs, us = self._splitz(z)
                
                J = 0
                for i in range(self.N):
                    x = xs[i*self.statedim:(i+1)*self.statedim]
                    u = us[i*self.inputdim:(i+1)*self.inputdim]
                    J += self.ocp.L(x, u)
                    
                xN = xs[-(self.statedim):]
                J += self.ocp.phi(xN)

                return J
            
            self._F = jit(_F)
            return self._F(z)

    
    def g(self, z):
        try:
            return self._g(z)
        except:
            self._g = jit(grad(self.F))
            return self._g(z)

    
    def c(self, z):
        try:
            return self._c(z)
        
        except:
            
            def _c(z):
                xs, us = self._splitz(z)
                
                _c = []
                
                for i in range(self.N):
                    x = xs[i*self.statedim:(i+1)*self.statedim]
                    xnext = xs[(i+1)*self.statedim:(i+2)*self.statedim]
                    u = us[i*self.inputdim:(i+1)*self.inputdim]
                    
                    _c.append(xnext - x - self.collocation_fn(x, u, self.dt))

                if 'x0' in self.bc:
                    x0 = xs[:self.statedim]
                    _c.append(x0 - self.bc['x0'])
                if 'xN' in self.bc:
                    xN = xs[-self.statedim:]
                    _c.append(xN - self.bc['xN'])
                if 'u0' in self.bc:
                    u0 = us[:self.inputdim]
                    _c.append(u0 - self.bc['u0'])
                if 'uN' in self.bc:
                    uN = us[-self.inputdim:]
                    _c.append(uN - self.bc['uN'])

                return jnp.concatenate(_c)

            self._c = jit(_c)
            return self._c(z)
            

    def G(self, z):
        """
        Returns LinearOperator of Jacobian of c where matvec is jvp
        """

        # Jacobian-vector product
        @jit
        def matvec(v):
            return jvp(self.c, (z,), (v,))[1]

        # Vector-Jacobian product
        @jit
        def rmatvec(v):
            _, vjp_fn = vjp(self.c, z)
            return vjp_fn(v)[0]

        shape = (self.nconstraints, self.nvars)
        return LinearOperator(shape, matvec=matvec, rmatvec=rmatvec, dtype=jnp.float32)
    

    def H(self, z, lam):

        # Gradient of Lagrangian w.r.t. z
        @jit
        def G_L(z):
            return self.g(z) - self.G(z).T @ lam

        # Hessian-vector product
        @jit
        def matvec(v):
            return grad(lambda x: jnp.vdot(G_L(x), v))(z)

        @jit
        def rmatvec(v):
            return matvec(v).T
        
        n = self.nvars
        return LinearOperator((n,n), matvec=matvec, dtype=jnp.float32)
        

    def KKT(self, z, lam):

        n, m = self.nvars, self.nconstraints

        G = self.G(z)
        H = self.H(z, lam)

        @jit
        def matvec(v):

            v1 = v[:n]
            v2 = v[n:]

            res1 = (H @ v1) + (G.T @ v2)
            res2 = G @ v1

            return jnp.concatenate([res1, res2])

        @jit
        def rmatvec(v):
            return matvec(v).T

        return LinearOperator((n+m,n+m), matvec=matvec, rmatvec=rmatvec, dtype=jnp.float32)


    def plot(self, z):
        
        xs, us = self._splitz(z)
        xs = list(xs.reshape(-1, self.statedim))
        us = list(us.reshape(-1, self.inputdim))

        for i in range(self.N):
            xs[i+1] = xs[i] + self.collocation_fn(xs[i], us[i])

        fig = plt.figure()
        self.ax = None

        self.artist = None
        
        def func(x):        
            self.artist = self.ocp.render(x, fig, self.artist)
            return self.artist,

        anim = animation.FuncAnimation(fig, func, frames=xs, blit=True, interval=200)
        plt.close(fig)
        
        return anim
        
    
def solve(problem: NlpProblem, x0=None, solver=gmres, precond_fn=None, eps=1e-4, max_iters=1000, outer_callback=None, verbose=True):
    """Solves a nonlinear programming problem by solving the KKT system

    Args:
        problem (NlpProblem): Instantiated NLP problem
        x0 (ndarray): Initial guess for solution of NLP problem (default None)
        solver (callable): Function that solves the KKT system (default gmres)
        precond_fn (callable): Function that generates a preconditioner for the KKT system (default None)
        eps (float): Tolerance for stopping condition concerning norm of solution (default 1e-4)
        max_iters (int): Maximum number of newton method iterations (default 1000)
        outer_callback (callable): Callback on newton method iterations (default None)
        verbose (bool): Verbosity on/off (default False)
    Returns:
        xstar (ndarray): Optimized solution to NLP problem
    """

    m, n = problem.nconstraints, problem.nvars

    if x0 is None:
        x0 = jnp.zeros(n)

    # Init arrays
    x = jnp.array(x0)
    lam = jnp.zeros(m)
    z = jnp.zeros(n+m)

    for it in range(max_iters):

        if verbose:
            print(f"\n--iter: {it+1}")
            
        gx, cx = problem.g(x), problem.c(x)

        K = problem.KKT(x, lam)

        b = jnp.block([gx, cx])

        if precond_fn:
            M = precond_fn(problem, x, lam)
        else:
            M = None

        z = solver(K, b, stop_gradient(z), M=M)[0]

        p = -z[:n]
        lam = z[n:]

        # TODO - line search
        x += p

        if outer_callback:
            outer_callback(x, lam)
        
        if jnp.linalg.norm(p) < eps and jnp.linalg.norm(cx) < eps:
            break
    
    if verbose:
        print(f"Optimized in {it+1} iterations")

    return x
