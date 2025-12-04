import numpy as np
class Game():
  def __init__(self, hawks, pigeons, target, dt, capture_radius=10.0):
    self.hawks = hawks
    self.pigeons = pigeons
    self.target = target
    self.dt = dt
    self.time = 0.0
    self.capture_radius = capture_radius

    self.trajectories = {
      "hawks": [[] for _ in range(len(hawks))],
      "pigeons": [[] for _ in range(len(pigeons))],
    }

    # Statut : 1 = actif, 0 = capturé
    self.pigeon_alive = [1] * len(pigeons)
    self.hawk_alive = [1] * len(hawks)


  def check_capture(self):
      for hi, hawk in enumerate(self.hawks):
          if not self.hawk_alive[hi]:
              continue

          hp = hawk.state[:3]

          for pi, pigeon in enumerate(self.pigeons):
              if not self.pigeon_alive[pi]:
                  continue

              pp = pigeon.state[:3]

              d = np.linalg.norm(hp - pp)

              if d <= self.capture_radius:
                  self.pigeon_alive[pi] = 0
                  print(f"[{self.time:.2f}s] Hawk {hi} captured pigeon {pi}")

  def check_target_capture(self):
    for pi, pigeon in enumerate(self.pigeons):
      if not self.pigeon_alive[pi]:
        continue

      pp = pigeon.state[:3]
      d = np.linalg.norm(pp - self.target)

      if d <= self.capture_radius:
        print(f"[{self.time:.2f}s] Pigeon {pi} captured the target!")
        return True  # game over for hawks

    return False



  def update(self):
    for i, pigeon in enumerate(self.pigeons):
      u = pigeon.control(
        self.target,
        self.hawks,
        self.pigeons
      )
      pigeon.step(u, self.dt)
      self.trajectories["pigeons"][i].append(pigeon.state[:3].copy())

    for i, hawk in enumerate(self.hawks):
      u = hawk.control(
        self.pigeons
      )
      hawk.step(u, self.dt)
      self.trajectories["hawks"][i].append(hawk.state[:3].copy())

    self.check_capture()

    if self.check_target_capture():
      print(f"one pigeon reached the target at t={self.time:.2f}s")
      return "TARGET_CAPTURED"

    if sum(self.pigeon_alive) == 0:
      print(f"All pigeons captured at t={self.time:.2f}s")
      return "ALL_PIGEONS_CAPTURED"

    self.time += self.dt
    return None

  def run(self, T):
    steps = int(T / self.dt)
    for _ in range(steps):
      status = self.update()
      if status is not None:
        return status

    return "TIME_LIMIT_REACHED"
