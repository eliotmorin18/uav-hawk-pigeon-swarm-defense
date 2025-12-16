import numpy as np

class Game():
  def __init__(self, hawks, pigeons, target, dt, capture_radius):
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


  def check_capture(self):
      for hi, hawk in enumerate(self.hawks):

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

    # ---------- 1. MOVE PIGEONS ----------
    for pi, pigeon in enumerate(self.pigeons):
      if not self.pigeon_alive[pi]:
            continue

      u = pigeon.control(
        self.target,
        self.hawks,
        self.pigeons
      )
      pigeon.apply_acceleration_control(u)
      pigeon.step(self.dt)
      self.trajectories["pigeons"][pi].append(pigeon.state[:3].copy())

    # ---------- 2. MOVE HAWKS ----------
    alive_pigeons = [p for pi, p in enumerate(self.pigeons) if self.pigeon_alive[pi]]
    for hi, hawk in enumerate(self.hawks):

      # Check if there are at least one visible pigeons in the Rs
      visible_pigeons = [
        p for p in alive_pigeons
        if np.linalg.norm(p.state[:3] - hawk.state[:3]) <= hawk.sensing_radius
      ]

      if visible_pigeons :
        hawk.current_target = hawk.choose_target(alive_pigeons)
        u = hawk.control(hawk.current_target)
      else:
        hawk.current_target = None
        u = np.zeros(3)

      hawk.apply_acceleration_control(u)
      hawk.step(self.dt)
      self.trajectories["hawks"][hi].append(hawk.state[:3].copy())

    # ---------- 3. CHECK CAPTURES ----------
    self.check_capture()

    # ---------- 4. TERMINAL CONDITIONS ----------
    if all(alive == 0 for alive in self.pigeon_alive):
        print(f"[{self.time:.2f}s] All pigeons have been captured.")
        return "ALL_PIGEONS_CAPTURED"

    if self.check_target_capture():
        print(f"[{self.time:.2f}s] A pigeon reached the target.")
        return "TARGET_CAPTURED"

    # ---------- 5. TIME UPDATE ----------
    self.time += self.dt
    return None


  def run(self, T):
      steps = int(T / self.dt)
      for step_num in range(steps):
          status = self.update()

          
      #if step_num <=50:
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
