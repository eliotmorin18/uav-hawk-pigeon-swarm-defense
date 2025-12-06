import numpy as np
import json
import numpy as np
from models.hawks import Hawk
from models.pigeon import Pigeon
from core.game import Game

# Positions initiales
hawk_pos_init = [0.0, 50.0, 100.0]
pigeon_pos = [-200.0, 100.0, 100.0]

# Direction vers le pigeon
direction = np.array(pigeon_pos) - np.array(hawk_pos_init)
r = np.linalg.norm(direction)

print(f"Direction: {direction}")
print(f"Distance r: {r}")

mu = np.arcsin(np.clip(direction[2] / r, -1, 1))
phi = np.arctan2(direction[1], direction[0])

print(f"mu={np.degrees(mu):.1f}°")
print(f"phi={np.degrees(phi):.1f}°")

vz = 400 * np.sin(mu)
print(f"vz initial: {vz}")