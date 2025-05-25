# Real-World SAC Robot Training (RL-2025)

This repository contains a real-world Soft Actor-Critic (SAC) reinforcement learning setup using:

* 🯞 **UR robot** (via RTDE interface)
* ✊ **Robotis PRO Series Gripper**
* 🔍 **FSR force sensors (2-channel analog readout)**
* 💻 All controlled directly from a single machine (no ZMQ)

---

## 📁 Clone Instructions

> ⚠️ **IMPORTANT:** When cloning this repository, rename the folder to `RL_2025` to match internal module paths.

```bash
git clone https://github.com/omletkang/RL-2025.git RL_2025
cd RL_2025
```

---

## 📦 Structure

```
RL_2025/
├── robot/               # Hardware interface modules
│   ├── gripper.py
│   ├── ur_robot.py
│   └── fsr_sensor.py
├── rollout.py           # Real-world environment wrapper
├── train_sac.py         # Training script using SAC
├── sac.py               # Soft Actor-Critic implementation
├── run/                 # Automatically created for storing logs and checkpoints
└── README.md
```

---

## 🚀 Quick Start

Train SAC on the real robot for 100 episodes:

```bash
python train_sac.py --n_episodes 100
```

To resume from a previous run:

```bash
python train_sac.py --resume run/2025_05_25_1530
```

---

## 🧠 Observation and Action

* **Observation**: `[TCP z-height, gripper_pos, FSR A0, FSR A1]`
* **Action**: `Gripper position` (normalized -1 to 1, mapped to 0–550 internally)

---

## 📜 License

This project is developed in a research setting. Please contact the author for reuse or collaboration.

---

## 👤 Author

Seung Hoon Kang (Soft Robotics and Bionics Lab, Seoul National University)
[GitHub: omletkang](https://github.com/omletkang)
