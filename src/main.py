import numpy as np
import matplotlib.pyplot as plt

from core.dynamics import rk4_step
from core.converter import control_converter
from models.uav import UAV

def main():
    dt = 0.1
    N = 1000

    # Initial state: [x, y, z, V, mu, phi]
    uav = UAV()

    uav.controls = {
            "Vdot": 0.0,
            "mudot": 0.02,  # Slight climb
            "phidot": 0.00,
            "gamma": 0.1    # Small flight path angle
        }

    trajectory = np.zeros((N, 6))

    for i in range(N):
        trajectory[i, :] = uav.state
        uav.step(dt)

    # (Optional) Example using control_converter
    # u = np.array([2.0, 0.5, 0.1])
    # mu, phi = state[4], state[5]
    # nx, nf, gamma = control_converter(u, mu, phi)
    # controls = {'nx': nx, 'nf': nf, 'gamma': gamma}

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

    print("final state:", uav.state)


if __name__ == "__main__":
    main()