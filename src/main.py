import numpy as np
import matplotlib.pyplot as plt

from core.dynamics import rk4_step
from core.converter import control_converter

def main():
    # Initial state: [x, y, z, V, mu, phi]
    state = np.array([0., 0., 0., 20., 0.0, 0.0])
    dt = 0.01
    N = 1000
    trajectory = np.zeros((N, len(state)))

    # Example: simple climb with bank
    controls = {'Vdot': 0.0, 'mudot': 0.02, 'phidot': 0.0, 'gamma': np.deg2rad(20)}

    # (Optional) Example using control_converter
    # u = np.array([2.0, 0.5, 0.1])
    # mu, phi = state[4], state[5]
    # nx, nf, gamma = control_converter(u, mu, phi)
    # controls = {'nx': nx, 'nf': nf, 'gamma': gamma}

    for i in range(N):
        trajectory[i] = state
        state = rk4_step(state, controls, dt)

    # Plot trajectory
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='UAV path')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('3D UAV Trajectory')
    ax.legend()
    plt.show()

    print("final state:", state)


if __name__ == "__main__":
    main()