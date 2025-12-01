import numpy as np

g = 9.8

def control_converter(u, mu, phi):
    """
    Convert acceleration commands (u_x, u_y, u_z) from 2nd-order integrator
    into physical UAV control commands (n_x, n_f, gamma)
    following the Hawk–Pigeon Game paper.
    """

    ux, uy, uz = u

    # M matrix from the paper
    M = np.array([
    [np.cos(mu)*np.cos(phi), -np.sin(mu)*np.cos(phi), -np.cos(mu)*np.sin(phi)],
    [np.cos(mu)*np.sin(phi), -np.sin(mu)*np.sin(phi), np.cos(mu)*np.cos(phi)],
    [np.sin(mu), np.cos(mu), 0]
    ])

    # Invert M
    M_inv = np.linalg.inv(M)

    # Compute Γ
    Gamma = M_inv @ np.array([ux/g, uy/g, uz*np.cos(mu)/g]) + np.array([np.sin(mu), np.cos(mu), 0.0])

    nx = Gamma[0]
    gamma = np.arctan2(Gamma[2], Gamma[1])

    if np.abs(np.sin(gamma)) > 1e-6:
        nf = Gamma[2] / np.sin(gamma)
    else:
        nf = Gamma[1] / np.cos(gamma)

    return nx, nf, gamma