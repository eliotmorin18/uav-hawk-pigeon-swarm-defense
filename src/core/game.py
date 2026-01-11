import numpy as np

class Game():
  def __init__(self, hawks, pigeons, target, dt, capture_radius, experiment_mode):
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

    # hawk_index -> pigeon_index
    self.current_assignments = {}

    self.experiment_mode = experiment_mode


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

  def compute_pigeon_danger(self):
    pigeon_danger = {}
    for pi, pigeon in enumerate(self.pigeons):
      if not self.pigeon_alive[pi]:
        continue

      pp = pigeon.state[:3]
      t_target = np.linalg.norm(pp - self.target) / (pigeon.V + 1e-9)

      intercept_candidates = []

      for hi, hawk in enumerate(self.hawks):

        ph = hawk.state[:3]
        distance = np.linalg.norm(pp - ph) - self.capture_radius

        if distance <= 0.0:
          intercept_candidates = [(0.0, hi)]
          break

        Vh = hawk.V
        Vp = pigeon.V
        V_rel = Vh - Vp

        if V_rel <= 1e-6:
          continue

        t_intercept = distance / V_rel
        intercept_candidates.append((t_intercept, hi))

      if intercept_candidates:
        intercept_candidates.sort(key=lambda x: x[0])
        t_min = intercept_candidates[0][0]
        hawks_ranked = [hi for _, hi in intercept_candidates]
      else :
        t_min = np.inf
        hawks_ranked = []

      delta_t = t_min - t_target
      danger = (1.0 / (t_target + 1e-6)) * (1.0 / (1.0 + np.exp(delta_t / 2.0)))

      pigeon_danger[pi] = {
        "danger" : danger,
        "t_intercept" : t_min,
        "best_hawks" : hawks_ranked
      }

    return pigeon_danger

  def rank_pigeons_by_danger(self, pigeon_danger):
      """
      Return list of pigeon indices sorted by decreasing danger.
      """
      return sorted(
          pigeon_danger.keys(),
          key=lambda pi: pigeon_danger[pi]["danger"],
          reverse=True
      )

  def assign_hawks_to_pigeons(self, ranked_pigeons, pigeon_danger, hysteresis_ratio = 0.25):
    """
    Assign hawks to pigeons based on danger ranking.
    Returns a dict: hawk_index -> pigeon_index
    """

    assignments = {}

    available_hawks = set(range(len(self.hawks)))

    # keep the former targets if they are available
    for hi, pi in self.current_assignments.items():
        if hi not in available_hawks:
            continue
        if pi not in pigeon_danger:
            continue

        assignments[hi] = pi
        available_hawks.remove(hi)

    # New allocation with fallback + hysteresis
    for pi in ranked_pigeons:
      if not available_hawks:
        break

      for hi in pigeon_danger[pi]["best_hawks"]:
        if hi not in available_hawks:
                continue

        # Hysteresis : we are comparing with the current target
        if hi in self.current_assignments:
          old_pi = self.current_assignments[hi]
          old_danger = pigeon_danger.get(old_pi, {}).get("danger", 0.0)
          new_danger = pigeon_danger[pi]["danger"]

          if new_danger < old_danger * (1.0 + hysteresis_ratio):
            continue

        assignments[hi] = pi
        available_hawks.remove(hi)
        break  # pigeon assigned, next pigeon

    self.current_assignments = assignments
    return assignments

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

    # ---------- 2.a GLOBAL HAWK–PIGEON ASSIGNMENT ----------
    if self.experiment_mode == "full":

        pigeon_info = self.compute_pigeon_danger()
        ranked_pigeons = self.rank_pigeons_by_danger(pigeon_info)
        assignments = self.assign_hawks_to_pigeons(ranked_pigeons, pigeon_info)

        for hi, hawk in enumerate(self.hawks):
            if hi in assignments:
                pi = assignments[hi]
                hawk.set_target(self.pigeons[pi])
            else:
                hawk.set_target(None)

    else:
        # paper / paper_anticipation: no global assignment
        self.current_assignments = {}

    # ---------- 2. MOVE HAWKS ----------
    for hi, hawk in enumerate(self.hawks):
      # only alive pigeons should be considered in paper modes
      alive_pigeons = [p for pi, p in enumerate(self.pigeons) if self.pigeon_alive[pi]]

      target = hawk.choose_target(alive_pigeons, experiment_mode=self.experiment_mode)


      if target is not None:
        u = hawk.control(target, experiment_mode=self.experiment_mode)
      else:
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
