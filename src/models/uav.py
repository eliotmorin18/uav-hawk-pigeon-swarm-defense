import numpy as np
from core.dynamics import rk4_step  # Ton moteur physique
from core.converter import control_converter

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
            "nx": 0.0,     # longitudinal load factor
            "nf": 0.0,     # normal load factor
            "gamma": 0.0   # bank angle
        }

        p_hawk = self.state[:3]
        v_hawk = self.velocity_vector
        print(f"[INIT] p_hawk = {p_hawk}")
        print(f"[INIT] v_hawk = {v_hawk}")
        print(f"[INIT] V (speed) = {self.V}")
        print(f"[INIT] mu = {np.degrees(self.mu)}°")
        print(f"[INIT] phi = {np.degrees(self.phi)}°")

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


    def apply_acceleration_control(self, u):
        """
        Convert a 2nd-order acceleration vector u = [ux, uy, uz]
        into the physical control laws (nx, nf, gamma).

        This matches exactly the conversion described in the
        Hawk–Pigeon Game paper (equations (3)-(6)).
        """
        u = np.asarray(u, dtype=float)
        assert u.shape == (3,), "u must be a vector [ux, uy, uz]"

        ux, uy, uz = u
        nx, nf, gamma = control_converter([ux, uy, uz], self.mu, self.phi)

        # Update control dictionary for rk4_step()
        self.controls["nx"] = nx
        self.controls["nf"] = nf
        self.controls["gamma"] = gamma


    # ----------- Simulation step -----------

    def step(self, dt):
        """Advance the UAV by dt using rk4."""
        self.state = rk4_step(self.state, self.controls, dt)
        return self.state

    def __repr__(self):
        return (f"UAV(x={self.x:.1f}, y={self.y:.1f}, z={self.z:.1f}, "
                f"V={self.V:.1f}, mu={np.degrees(self.mu):.1f}°, "
                f"phi={np.degrees(self.phi):.1f}°)")
<<<<<<< HEAD



=======
>>>>>>> ea5c25f3a97fb094fe089d3a0eda02af62587bfa
