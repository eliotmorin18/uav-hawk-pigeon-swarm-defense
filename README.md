# Coordinated UAV Swarm Target Defense Based on a Hawk–Pigeon Game

## Overview

This repository contains the code and reports for a research project on **UAV swarm target defense**, inspired by the **Hawk–Pigeon Game**.  
The project builds upon the analytical framework introduced by Ruan et al. (2021) and progressively refines it to address practical limitations observed in multi-agent simulations.

Starting from the original bio-inspired model based on local target selection and proportional navigation, we introduce:

- a **predictive interception mechanism** to mitigate tail-chasing behavior,
- a **global danger-based coordination strategy** to avoid redundant pursuits and improve scalability.

While early project stages envisioned learning-based extensions (imitation learning and reinforcement learning), the final work focuses on a **robust, deterministic, and interpretable baseline**, which proved essential before any learning-based approach could be meaningfully applied.

All results presented in the final report are fully reproducible using the code provided in this repository.

---

## Reference Paper

This project is primarily based on:

> W. Ruan, Y. Sun, Y. Deng, and H. Duan,  
> _Hawk–pigeon game tactics for unmanned aerial vehicle swarm target defense_,  
> IEEE Transactions on Cybernetics, vol. 51, no. 9, pp. 4423–4436, 2021.  
> Hawk-Pigeon_Game_Tactics_for_Unmanned_Aerial_Vehicle_Swarm_Target_Defense.pdf

---

## Project Goals

- Reproduce the original Hawk–Pigeon swarm defense architecture.
- Analyze its limitations in multi-defender, multi-attacker scenarios.
- Improve interception efficiency through **predictive interception**.
- Design a **global coordination mechanism** based on threat (danger) assessment.
- Provide a clean, deterministic baseline suitable for future learning-based extensions.
- Produce reproducible simulations and a polished scientific report.

**Achieved level:**  
All deterministic objectives were successfully achieved and validated through controlled simulations. Learning-based extensions were deliberately postponed and are discussed as future work.

---

## Repository Structure

```text
src/
├── config/
│   └── parameters.json        # Simulation parameters and experiment mode
│
├── core/
│   ├── converter.py           # Control command to UAV dynamics conversion
│   ├── dynamics.py            # UAV motion and integration model
│   └── game.py                # Game logic and global coordination
│
├── models/
│   ├── uav.py                 # Base UAV class
│   ├── hawks.py               # Hawk (defender) behavior and control
│   └── pigeon.py              # Pigeon (attacker) behavior
│
├── results/
│   └── trajectoire.json       # Recorded trajectories (generated per run)
│
├── main.py                    # Entry point for simulations
├── plot_trajectoires.py       # 2D/3D trajectory visualization
└── plot_video.py              # Video generation from simulations
```

---

## Reports

This repository includes all three project reports:

- First report (first_report.pdf) – Analysis of the Hawk–Pigeon Game and baseline implementation

- Second report (second_report.pdf) – Intermediate extensions and exploratory learning directions

- Final report (final_report.pdf / final_report.tex) – Coordinated UAV swarm defense with predictive interception

The PDF of the final report corresponds exactly to the version submitted for the course.

---

## Installation

### Requirements

- Python ≥ 3.10
- Libraries: numpy, scipy, matplotlib, numba, pandas, seaborn
- Optionnal for future extensions : torch or tensorflow

### Setup

```bash
git clone https://github.com/eliotmorin18/uav-hawk-pigeon-swarm-defense.git
cd uav-hawk-pigeon-swarm-defense
pip install -r requirements.txt
```

---

## Running the Simulation

Simulation parameters (initial positions, number of hawks/pigeons, mode selection, time step) have to be defined in:

```bash
src/config/parameters.json
```

All simulations are launched from main.py.

```bash
python3 src/main.py
```

### Experiment Modes

The framework supports three experimental configurations:

- paper - Original Hawk–Pigeon model with local target selection and reactive pursuit.

- paper_anticipation - Local target selection combined with predictive interception.

- full - Predictive interception + global danger-based coordination.

The mode can be selected in parameters.json:

- "experiment_mode": "paper"
- "experiment_mode": "paper_anticipation"
- "experiment_mode": "full"

During execution, the simulation: records UAV trajectories, logs interception events, and saves data for post-processing and visualization in "trajectories.json"

### Generated plots

To generate plots (2D and 3D trajectories) you have to run "plot_trajectories.py"

### Generated videos

To generate videos of trajectories, you have to run "plot_video.py"

---

## Reproducibility

All figures and results presented in the final report were generated using:

the same initial conditions,

identical parameters across experiment modes,

independent simulation runs.

This ensures fair and reproducible comparisons between configurations.

---

## Team

- Eliot Morin — [@eliotmorin18](https://github.com/eliotmorin18)
- Hugo Trébert — [@hugotrbt](https://github.com/hugotrbt)
- Mikhaïl Iakovlev — [@miakovlevv](https://github.com/miakovlevv)
