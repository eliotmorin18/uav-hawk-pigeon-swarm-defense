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
    self.debug_step_count = 0


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

  """
  def print_status(self):
      print(f"\n--- T = {self.time:.2f}s ---")
      
      # Distances des pigeons à la target
      for pi, pigeon in enumerate(self.pigeons):
        if self.pigeon_alive[pi]:
          dist_to_target = np.linalg.norm(pigeon.state[:3] - self.target)
      
      # Hawks et leurs cibles
      for hi, hawk in enumerate(self.hawks):
        if self.hawk_alive[hi]:
          if hawk.current_target is not None:
            dist_to_target = np.linalg.norm(hawk.state[:3] - hawk.current_target.state[:3])
            target_idx = self.pigeons.index(hawk.current_target)
            print(f"  Hawk {hi}: target = Pigeon {target_idx}, distance = {dist_to_target:.1f}m")
          else:
            print(f"  Hawk {hi}: no target")
  """

  def update(self):
    self.debug_step_count += 1
    for i, pigeon in enumerate(self.pigeons):
      u = pigeon.control(
        self.target,
        self.hawks,
        self.pigeons
      )
      pigeon.apply_acceleration_control(u)
      pigeon.step(self.dt)
      self.trajectories["pigeons"][i].append(pigeon.state[:3].copy())

    for i, hawk in enumerate(self.hawks):
      # Chercher une cible parmi les pigeons vivants
      alive_pigeons = [p for pi, p in enumerate(self.pigeons) if self.pigeon_alive[pi]]
      
      if alive_pigeons:
        hawk.current_target = hawk.choose_target(alive_pigeons)
        u = hawk.control(hawk.current_target)
      else:
        u = np.zeros(3)

      hawk.apply_acceleration_control(u)
      hawk.step(self.dt)
      self.trajectories["hawks"][i].append(hawk.state[:3].copy())

      self.check_capture()

      if self.check_target_capture():
        print(f"one pigeon reached the target at t={self.time:.2f}s")
        return "TARGET_CAPTURED"

      if sum(self.pigeon_alive) == 0:
        print(f"All pigeons captured at t={self.time:.2f}s")
        return "ALL_PIGEONS_CAPTURED"

      self.time += self.dt
  
      if hawk.current_target:
        target_pos = hawk.current_target.state[:3]
        dist = np.linalg.norm(hawk.state[:3] - target_pos)
        direction_to_target = (target_pos - hawk.state[:3]) / np.linalg.norm(target_pos - hawk.state[:3])

    self.time += self.dt
    return None

  def run(self, T):
      steps = int(T / self.dt)
      for step_num in range(steps):
          status = self.update()

          
          if step_num <=50:
              print(f"[STEP {step_num}] hawk pos = {self.hawks[0].state[:3]}, distance = {np.linalg.norm(self.hawks[0].state[:3] - self.pigeons[0].state[:3]):.1f}m")
              print(f"[STEP {step_num}] pigeon pos = {self.pigeons[0].state[:3]}, distancetarget = {np.linalg.norm(self.pigeons[0].state[:3] - self.target[:3]):.1f}m")
              print(f"  V = {self.hawks[0].V:.2f} m/s")
              print(f"  mu = {np.degrees(self.hawks[0].mu):.2f}°")
              print(f"  phi = {np.degrees(self.hawks[0].phi):.2f}°")
              print(f"  v_velocity = {self.hawks[0].velocity_vector}")
              print(f"  controls: nx={self.hawks[0].controls['nx']:.3f}, nf={self.hawks[0].controls['nf']:.3f}, gamma={np.degrees(self.hawks[0].controls['gamma']):.2f}°")

          if status is not None:
              return status

      return "TIME_LIMIT_REACHED"
