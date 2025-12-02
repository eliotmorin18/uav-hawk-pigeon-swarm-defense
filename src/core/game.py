class Game():
  def __init__(self, hawks, pigeons, target, dt):
    self.hawks = hawks
    self.pigeons = pigeons
    self.target = target
    self.dt = dt
    self.time = 0.0
    
    self.trajectories = {
      "hawks": [[] for _ in range(len(hawks))],
      "pigeons": [[] for _ in range(len(pigeons))],
    }

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
      
    self.time += self.dt
    
  def run(self, T):
    steps = int(T / self.dt)
    for _ in range(steps):
      self.update()
      