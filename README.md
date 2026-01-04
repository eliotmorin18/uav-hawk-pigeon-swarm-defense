# UAV Swarm Target Defense

## Hawk–Pigeon Game with AI Extension

## Description

This project reproduces and **extends the hawk–pigeon game architecture** for **UAV swarm target defense**, inspired by the IEEE paper _Hawk–Pigeon Game Tactics for Unmanned Aerial Vehicle Swarm Target Defense_.

The goal is to build a **clean, controllable simulation environment** for studying:

- Swarm pursuit–evasion
- Cooperative interception
- Learning-based decision-making

Beyond reproduction, we introduce **practical coordination and control improvements** motivated by issues observed during implementation and simulation.

---

## Team

- Eliot Morin — [@eliotmorin18](https://github.com/eliotmorin18)
- Hugo Trébert — [@hugotrbt](https://github.com/hugotrbt)
- Mikhaïl Iakovlev — [@miakovlevv](https://github.com/miakovlevv)

---

## Scientific Starting Point

- _Hawk–Pigeon Game Tactics for Unmanned Aerial Vehicle Swarm Target Defense_
  IEEE Transactions on Industrial Informatics
  DOI: 10.1109/TII.2023.3248075

This repository **implements the full control pipeline** described in the paper and adapts it to a more robust and extensible simulation architecture.

---

## Implemented Core Features (Paper)

- 6-DOF fixed-wing UAV dynamic model
- Control command converter (second-order integrator → 6-DOF)
- Hawk-inspired pursuit:

  - PN + PP guidance with adaptive gains

- Pigeon-inspired attack and evasion:

  - Target attraction
  - Hawk avoidance
  - Inter-pigeon collision avoidance

- Soft capture with capture radius
- Metrics: win rate, captures, interception time

---

## Extensions and Modifications (Our Work)

### 1. Centralized Target Selection by Dangerosity (Game-Level)

**Key change from the paper**

Target selection is **no longer handled independently inside each hawk**.

Instead:

- Target selection is centralized in `Game`
- Pigeons are ranked by **dangerosity**, based on:

  - Distance to the protected target
  - Likely time-to-impact

- Hawks are assigned to pigeons **by priority**, ensuring:

  - No two hawks pursue the same pigeon
  - The most threatening pigeons are intercepted first

This replaces the original local “choose target” logic with a **global, consistent allocation strategy**.

---

### 2. Hawk–Hawk Coordination (No Redundant Pursuit)

- Explicit communication via shared game state
- One hawk → one pigeon at a time
- Automatic reassignment when:

  - A pigeon is captured
  - A target becomes unreachable or irrelevant

Result: cleaner swarm behavior and better interception efficiency.

---

### 3. Predictive Interception (Trajectory Anticipation)

- Time-to-go estimation under constant-velocity assumption
- Prediction of future pigeon position
- Guidance applied toward predicted interception point
- Fallback to classical PN if prediction is unstable

This fixes issues such as:

- Hawks continuing straight after capture
- Late or inefficient pursuit trajectories

---

### 4. Clear Capture and State Management

- Hawks never “die” — they only reassign targets
- Explicit capture state for pigeons
- Consistent capture radius handling across modules

This improves simulation stability and learning-readiness.

---

## AI / ML Orientation

The simulator is structured to easily generate datasets:

- Relative states (hawk–pigeon geometry)
- Assigned targets
- Control outputs
- Episode-level metrics

This enables:

- Supervised learning
- Imitation learning
- Future reinforcement learning experiments

---

## Project Goals

- Reproduce the reference hawk–pigeon model
- Fix practical coordination and control issues
- Prepare the system for learning-based extensions
- Use the simulator as a research and experimentation platform

---

## Roadmap (Next Steps)

- Learned target danger estimation
- Learned target assignment (MLP / attention models)
- Learned gain adaptation for PN/PP
- Attacker counter-strategies
- Partial observability and sensor noise
- Large-scale swarm experiments

---

## Getting Started

### Requirements

- Python ≥ 3.10
- `numpy`, `scipy`, `matplotlib`, `numba`
- `torch` or `tensorflow`
- `pandas`, `seaborn`

### Installation

```bash
git clone https://github.com/eliotmorin18/uav-hawk-pigeon-swarm-defense.git
cd uav-hawk-pigeon-swarm-defense
pip install -r requirements.txt
```
