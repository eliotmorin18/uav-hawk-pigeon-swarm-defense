import numpy as np

g = 9.8

def dynamics(state, controls) :
    """
    state: [x, y, z, V, mu, phi]
    controls: dict with either:
      - 'nx','ny','nf','gamma'  (direct load-factors + bank)
      OR
      - 'Vdot','mudot','phidot','gamma' (desired time-derivs, converted below)
    returns: state_dot as numpy array
    """

    x, y, z, V, mu, phi = state
    gamma = controls.get('gamma', 0.0)
    
    # If provided time-derivs, convert to load factors
    if 'Vdot' in controls and 'mudot' in controls and 'phidot' in controls:
        Vdot = controls['Vdot']
        mudot = controls['mudot']
        phidot = controls['phidot']

        nx = Vdot / g + np.sin(mu)

        # protect cos(gamma) near zero
        cosg = np.cos(gamma)
        if abs(cosg) < 1e-6:
            ny = np.sign(cosg) * 1e6  # large value - indicates singular control request
        else:
            ny = (mudot * V / g + np.cos(mu)) / cosg

        sing = np.sin(gamma)
        if abs(sing) < 1e-6:
            nf = np.sign(sing) * 1e6
        else:
            nf = (phidot * V * np.cos(mu)) / (g * sing)

    else:
        nx = controls.get('nx', 0.0)
        ny = controls.get('ny', 0.0)
        nf = controls.get('nf', 0.0)

    # State derivatives
    x_dot = V * np.cos(mu) * np.cos(phi)
    y_dot = V * np.cos(mu) * np.sin(phi)
    z_dot = V * np.sin(mu)

    V_dot = g * (nx - np.sin(mu))
    mu_dot = g * (ny * np.cos(gamma) - np.cos(mu)) / max(V, 1e-6)

    denom = V * np.cos(mu)
    if abs(denom) < 1e-6:
        phi_dot = 0.0
    else:
        phi_dot = g * nf * np.sin(gamma) / denom

    return np.array([x_dot, y_dot, z_dot, V_dot, mu_dot, phi_dot])
  
def rk4_step(state, controls, dt):
    k1 = dynamics(state, controls)
    k2 = dynamics(state + 0.5*dt*k1, controls)
    k3 = dynamics(state + 0.5*dt*k2, controls)
    k4 = dynamics(state + dt*k3, controls)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)