# 🏆 FlappyAlpha: Reinforcement Learning vs Human

## 🎮 Overview

**FlappyAlpha** is a reinforcement learning (RL) project inspired by AlphaGo, where an RL agent learns to play a custom Flappy Bird game environment with realistic physics (gravity, collision, etc.). The agent (FlappyAlpha) is trained using Q-Learning to maximize its score by passing obstacles and avoiding crashes. The project also features human vs. agent battles to showcase the agent's learning progress.

## 🎮 Features

- Custom Flappy Bird environment built with Pygame
- Q-Learning agent with reward/punishment system
- RL training for "Beginner" and "Hard" agents
- Human vs Agent battle playground with score visualization
- Performance interpretation and result plots

## 🎮 Tech Stack

- **Game & UI:** Python, Pygame
- **Reinforcement Learning:** Q-Learning (NumPy)
- **Data Analysis & Result Visualization:** NumPy, Matplotlib

## 🎮 Simple Demo
![Demo video](<assets/Screen Recording 2025-02-23 170420.gif>)

## 🎮 Project Structure

```
rl_flappyBird/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/                        # Core game environment
│   └── flappy_bird.py
│
├── train/                      # Training scripts
│   ├── train_qLearningBEGINNER.py
│   └── train_qLearningHARD.py
│
├── models/                     # Trained models
│   ├── best_q_table_beginner.npy
│   └── best_q_table_hard.npy
│
├── results/                    # Result plots and analysis
│   ├── beginner_mode_scores.png
│   ├── hard_mode_scores.png
│   ├── user_vs_agent_scores.png
│   └── user_vs_agent_scoresHARD.png
│
├── playground/                 # Human vs agent
│   ├── playground.py
│   └── playgroundHARD.py
│
└── assets/                     # Assets
```

## 🎮 How to Use
> U can use virtual environment (venv), but since the dependencies are just common and less, so depend on yourself.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the RL Agent

> **Note:** Pre-trained Q-tables are provided in the `models/` folder. You can retrain if you wish.

- Train Beginner mode agent:
  ```bash
  python train/train_qLearningBEGINNER.py
  ```
- Train Hard mode agent:
  ```bash
  python train/train_qLearningHARD.py
  ```

### 3. Human vs. Agent Battle

- Play against Beginner agent:
  ```bash
  python playground/playground.py
  ```
- Play against Hard agent:
  ```bash
  python playground/playgroundHARD.py
  ```

### 4. View Results

- Training and battle results are auto saved as `.png` plots in the `results/` folder.

## 🎮 Highlights & Insights

- The RL agent receives positive rewards for passing pipes/blocks and negative penalties for crashing.
- As training episodes increase, the agent's performance improves dramatically:
  - After 2500 episodes `(beginner mode)`: Best score 17.
  - After 3000 episodes `(hard mode)`: Best score 75 (441% increase).
  - After 4000 episodes: Best score 3693 (near-perfect play).
  > U can modify the training parameter, ex. EPISODES = 4000 if u want.
- Human players can easily beat the Beginner agent `(2500 episodes)`, but the Hard agent `(3000 episodes)` is a tough opponent, often outperforming humans in average score, and I'm sure for `> 4000 episodes` training, human can't beat RL agent as it plays near perfect.

## 🎮 Acknowledgements

Inspired by AlphaGo and the power of reinforcement learning.

---

<div align="center">
  <p>💎 <em>Feel free to fork, do experiment, and challenge FlappyAlpha.</em></p>
</div>