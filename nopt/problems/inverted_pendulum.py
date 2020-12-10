import matplotlib.pyplot as plt
from nopt import OptControlProblem
import jax.numpy as jnp


class InvertedPendulum(OptControlProblem):
    """Inverted pendulum problem

    Dynamics taken from:
      Li, Weiwei, and Emanuel Todorov. "Iterative linear quadratic 
      regulator design for nonlinear biological movement systems." 
      ICINCO (1). 2004.
    """
    def __init__(self, N: int=10, h: float=1e-5):
        
        self.grav = 9.8
        self.m = self.l = 1
        self.mu = 0.01

        self.Q = jnp.eye(2)
        self.r = 1e-4
        
        
    def f(self, x, u):
        """
        Args:
           x: [theta, theta_dot]
           u: [torque]
        """
        x1, x2 = x
        u = u[0]

        d = self.m * self.l**2
        theta_dot = x2
        theta_ddot = self.grav / self.l * jnp.sin(x1) - self.mu / d * x2 + 1./d * u
        return jnp.array([theta_dot, theta_ddot])
    

    def L(self, x, u):
        return 0.5 * (self.r * u.T @ u + x.T @ (self.Q @ x))


    def phi(self, xN):
        xN1, xN2 = xN
        return 0.5 * (10 * xN1**2 + xN2**2)

    def render(self, x, fig, artist=None):

        if artist is None:
            ax = fig.add_subplot(111)
            ax.set_xlim([-1.1, 1.1])
            ax.set_ylim([-1.1, 1.1])
            
        theta = -x[0] + jnp.pi/2

        x = jnp.array([0, jnp.cos(theta)])
        y = jnp.array([0, jnp.sin(theta)])

        if artist:
            artist.set_data(x, y)
        else:
            artist, = ax.plot(x, y)

        return artist
        
    
    @property
    def statedim(self):
        return 2

    @property
    def inputdim(self):
        return 1
    

        
