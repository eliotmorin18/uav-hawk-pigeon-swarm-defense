import numpy as np
from core.dynamics import rk4_step, dynamics  # Ton moteur physique

class UAV:
    """
    UAV 6-DOF class.
    Acts as a state container + helper utilities.
    Dynamics are computed in core/dynamics.py
    """

    def __init__(self, x=0., y=0., z=0., V=100., mu=0., phi=0.):
        self.state = np.array([x, y, z, V, mu, phi], dtype=float)

        # Default controls
        self.controls = {
            "Vdot": 0.0,
            "mudot": 0.0,
            "phidot": 0.0,
            "gamma": 0.0
        }

    # ----------- Properties -----------

    @property
    def x(self): return self.state[0]
    @x.setter
    def x(self, value): self.state[0] = value

    @property
    def y(self): return self.state[1]
    @y.setter
    def y(self, value): self.state[1] = value

    @property
    def z(self): return self.state[2]
    @z.setter
    def z(self, value): self.state[2] = value

    @property
    def V(self): return self.state[3]
    @property
    def mu(self): return self.state[4]
    @property
    def phi(self): return self.state[5]

    @property
    def velocity_vector(self):
        """Compute vx, vy, vz from state."""
        V, mu, phi = self.V, self.mu, self.phi
        vx = V * np.cos(mu) * np.cos(phi)
        vy = V * np.cos(mu) * np.sin(phi)
        vz = V * np.sin(mu)
        return np.array([vx, vy, vz])

    # ----------- Simulation step -----------

    def step(self, dt):
        """Advance the UAV by dt using rk4."""
        self.state = rk4_step(self.state, self.controls, dt)
        return self.state

    def __repr__(self):
        return (f"UAV(x={self.x:.1f}, y={self.y:.1f}, z={self.z:.1f}, "
                f"V={self.V:.1f}, mu={np.degrees(self.mu):.1f}°, "
                f"phi={np.degrees(self.phi):.1f}°)")
