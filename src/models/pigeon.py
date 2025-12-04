import numpy as np
from models.uav import UAV

class Pigeon(UAV) :
  def __init__(self, x, y, z, V, mu, phi, params):
    super().__init__(x, y, z, V, mu, phi)
    self.k1 = params["k1"]   # attraction gain
    self.Re = params["Re"]   #pigeon's escaping safety radius
    self.Ra = params["Ra"]   #pigeon's avoiding collision safety radius

  def compute_attack_accel(self, target_position):
    """
    Equation (19) of the paper :
    u_attack = k1 * (pT - pp) / ||pT - pp||
    """

    p = self.state[0:3]
    target_position = np.asarray(target_position, dtype=float)

    direction = target_position - p
    dist = np.linalg.norm(direction)

    if dist < 1e-6 :
      return np.zeros(3)

    return self.k1 * direction / dist


  def compute_escape_accel(self, hawks):
    """
    Compute the escape acceleration component as defined in equation (20)
    of the paper

    This term models the pigeon’s tendency to flee from nearby hawks.
    For each hawk j located within the escape radius RE, an escape
    contribution is added in the opposite direction of the hawk.

    The escape acceleration is:

        u_escape = Σ_{j ∈ E2}  k2[j] * (p_pigeon - p_hawk[j]) / ||p_pigeon - p_hawk[j]||

    where the gain k2[j] is:

        k2[j] = exp( 1 + ( ||p_pigeon - p_hawk[j]|| - RE )² / RE² )

    and the escape set E2 is the set of hawks within RE:

        E2 = { j  |  ||p_pigeon - p_hawk[j]|| ≤ RE }

    Returns
    -------
    np.ndarray
        A 3D escape acceleration vector (shape (3,)).
    """
    
    p = self.state[:3]
    Re = self.Re
    uescape = np.zeros(3)

    for hawk in hawks :
      ph = hawk.state[:3]
      direction = p - ph
      dist = np.linalg.norm(direction)

      if dist < Re and dist > 1e-6 :
        k2 = np.exp(1 + ((dist - Re)**2 / Re**2) )
        uescape += k2 * direction / dist

    return uescape

  def compute_avoid_accel(self, pigeons):
    """
    Compute the collision-avoidance acceleration component as defined in
    equation (21) of the paper.

    This term models the pigeon’s tendency to avoid collisions with nearby
    pigeons. For each neighboring pigeon j located within the avoidance
    radius RA, a repulsive contribution is added in the direction pointing
    away from that pigeon.

    The avoidance acceleration is:

        u_avoid = Σ_{j ∈ E3}  k3[j] * (p_pigeon - p_pigeon[j]) / ||p_pigeon - p_pigeon[j]||

    where the gain k3[j] is:

        k3[j] = (RA - ||p_pigeon - p_pigeon[j]||) / RA

    and the avoidance set E3 is:

        E3 = { j  |  ||p_pigeon - p_pigeon[j]|| ≤ RA }

    Returns
    -------
    np.ndarray
        A 3D avoidance acceleration vector (shape (3,)).
    """
    p = self.state[:3]
    Ra = self.Ra
    uavoid = np.zeros(3)

    for pigeon in pigeons :
      if pigeon is self:
        continue

      pp = pigeon.state[:3]
      direction = p - pp
      dist = np.linalg.norm(direction)

      if dist < Ra and dist > 1e-6 :
        k3 = (Ra - dist) / Ra
        uavoid += k3 * direction / dist

    return uavoid

  def control(self, target_position, hawks, pigeons):
        u_attack = self.compute_attack_accel(target_position)
        u_escape = self.compute_escape_accel(hawks)
        u_avoid  = self.compute_avoid_accel(pigeons)

        return u_attack + u_escape + u_avoid

  def __repr__(self):
      p = self.state[:3]
      return f"Pigeon(pos=[{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}])"
