# UAV Swarm Target Defense – Hawk–Pigeon Game with AI Extension

## Description

This project studies and extends the **hawk–pigeon game architecture** for **UAV swarm target defense**, integrating **Artificial Intelligence (AI)** and **Machine Learning (ML)** techniques to enhance decision-making and adaptability.  
The classical model relies on a **6-degree-of-freedom (6-DOF) UAV system** with a **control command converter** that transforms a second-order integrator model into a realistic 6-DOF dynamic model. The defenders use **pursuit strategies inspired by hawk hunting**, while the attackers follow **pigeon-like group homing behavior**.

In our extended version, we apply **data-driven learning** to improve **coordination, interception efficiency, and predictive modeling**. Using **MLP (Multi-Layer Perceptron)** or **CNN (Convolutional Neural Network)** architectures, the UAV swarm learns from simulation data to adapt pursuit and evasion strategies in real time.  
We also perform **data analysis** on simulation metrics (win rate, number of captures, interception time) to identify optimal behaviors and configurations.

## Team

- Eliot Morin — [@eliotmorin18](https://github.com/eliotmorin18)
- Hugo Trébert — [@hugotrbt](https://github.com/hugotrbt)
- Mikhaïl Iakovlev — [@github-link](https://github.com/)

## Starting Point

- _Hawk–Pigeon Game Tactics for Unmanned Aerial Vehicle Swarm Target Defense_  
  (IEEE Transactions on Systems, Man, and Cybernetics: Systems, DOI: [10.1109/TSMC.2021.3125587](https://doi.org/10.1109/TSMC.2021.3125587))

## Project Goals

- Reproduce and extend the hawk–pigeon defense architecture.
- Implement a **6-DOF UAV model** and control command converter.
- Design **ML algorithms** (MLP/CNN) to enhance swarm coordination.
- Analyze simulation data to identify efficient strategies.
- Evaluate performance using **win rate**, **number of captures**, and **activity duration**.

## Plan and Milestones

- **Milestone 1 – Report 1 (2025-11-15):** In-depth reading of the paper, understanding the 6-DOF model, implementing a minimal simulator, and generating the first training data.
- **Milestone 2 – Report 2 (2025-12-15):** Implementation of MLP/CNN models for defense/attack strategy learning, reproduction of isochrones, and visualization.
- **Milestone 3 – Report 3 (2026-01-15):** Large-scale experiments, performance analysis, ablation studies, distributed extensions, and final report.

## Progress Tracking

- Managed via **GitHub Issues** and a **Project Board (Kanban)**.
- Weekly updates to the README.
- Commit conventions:
  - `feat:` new feature
  - `fix:` bug fix
  - `ml:` machine learning experiment/model
  - `docs:` documentation
  - `exp:` experiment or simulation
  - `refactor:` code improvement

## Getting Started

### Requirements

- Python ≥ 3.10
- Libraries: `numpy`, `scipy`, `matplotlib`, `numba`, `torch` or `tensorflow`, `pandas`, `seaborn`

### Installation

```bash
git clone https://github.com/eliotmorin18/uav-hawk-pigeon-swarm-defense.git
cd uav-hawk-pigeon-swarm-defense
pip install -r requirements.txt
```
