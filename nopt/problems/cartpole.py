from nopt import OptControlProblem
import jax.numpy as jnp


class CartPole(OptControlProblem):
    """Cart pole problem
    
    Dynamics taken from:
        https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-832-underactuated-robotics-spring-2009/readings/MIT6_832s09_read_ch03.pdf

    Args:
        mc (float): Mass of cart
        mp (float): Mass of pole
        l (float): Length of pole
    """

    def __init__(self, mc: float=2, mp: float=1, l: float=1):

        self.grav = 9.8
        self.mc = mc
        self.mp = mp
        self.m = (mc + mp)
        self.l = l

    def f(self, x, u):
        """
        Args:
            x: [x, xdot, theta, thetadot]
            u: [horizontal_force]
        """

        x, xdot, theta, thetadot = x
        F = u[0]

        s, c = jnp.sin(theta), jnp.cos(theta)

        mc, mp, m = self.mc, self.mp, self.m
        g, l = self.grav, self.l
        denom = mc + mp * (s**2)
        
        xddot = (F + mp * s * (l * thetadot**2 + g * c)) / (mc + mp * (s**2))
        thetaddot = (-F * c - mp * l * thetadot**2 * c * s - m * g * s)/(l * (mc + mp * (s**2)))

        return jnp.array([xdot, xddot, thetadot, thetaddot])

    def L(self, _, u):
        return 0.5 * u.T @ u

    def phi(self, _):
        return 0

    def render(self, x, fig, artist=None):

        if artist is None:
            ax = fig.add_subplot(111)

            lim = 3 * self.l
            ax.set_xlim([-lim, lim])
            ax.set_ylim([-lim, lim])

            self._lim = lim

        x, _, theta, _ = x
        
        if 3*abs(x) > self._lim:
            lim = 3*abs(x)
            ax.set_xlim([-lim, lim])
            ax.set_ylim([-lim, lim])
            self._lim = lim
        
        joint_x, joint_y = [x, 0]
        tip_x = joint_x + self.l * jnp.sin(theta)
        tip_y = -self.l * jnp.cos(theta)
        
        xs = [joint_x, tip_x]
        ys = [joint_y, tip_y]
        
        if artist:
            artist.set_data(xs, ys)
        else:
            artist, = ax.plot(xs, ys)
            
        return artist

    @property
    def statedim(self):
        return 4

    @property
    def inputdim(self):
        return 1

    
        
        
