<div align="center">

# Bomberman Reinforcement Learning

</div>
<div align="center">
  <img src="images/bomberman.png" width="500" alt="Bomberman">
</div>

A project for the course:  
*Reinforcement learning and autonomous agents*  

---

## Introduction
This project explores **Reinforcement Learning (RL)** techniques in the **Bomberman** environment. Our primary goal is to develop an intelligent agent capable of strategic decision-making, optimizing movement, bomb placement, and opponent avoidance.

## Environment Overview

The **Bomberman environment** is a **grid-based game** where agents navigate a map, place bombs, and attempt to maximize their score while avoiding self-destruction. Key mechanics include:

- 🏃 **Movement:** Agents can move in four cardinal directions within a structured grid.
- 💣 **Bomb Placement & Explosions:** Bombs detonate after a timer, destroying adjacent tiles and eliminating nearby agents.
- 📦 **Crates:** Breakable obstacles that clear paths when destroyed.
- 💰 **Coins:** Collectible items that contribute to the agent's score.
- ⚔️ **Opponent Interactions:** Agents must strategically evade or attack opponents to maximize survival chances.

Each agent in the environment follows a predefined structure and must be placed within the `agent_code` folder. An agent must contain a `callbacks.py` file, which includes the following required functions:

- **setup(self):** Initializes the agent and prepares it for gameplay.
- **act(self, game_state: dict) -> str:** Determines the agent's next move based on the game state.

For trainable agents, an additional `train.py` file is required, which should define the following functions:

- **game_events_occurred(self, old_game_state, self_action, new_game_state, events):** Handles intermediate rewards and updates the learning model.
- **end_of_round(self, last_game_state, last_action, events):** Processes final rewards and training updates at the end of each round.
- **reward_from_events(self, events):** Computes the total reward from all game events.

If a trainable agent does not include `train.py`, an exception will be raised.


## Agents Implemented

### 🔹 PPO-Based Agent
- **Algorithm:** Proximal Policy Optimization (PPO)
- **Status:** Currently being optimized
- **Location:** `agent_code/ppo`
- **Description:** Nucleus leverages PPO for policy learning. However, performance is still being refined to mitigate suboptimal behaviors like staying idle or failing to escape bomb blasts.

### 🔹 QL_Agent (Q-Learning Baseline)
- **Algorithm:** Q-Learning
- **Status:** Stable performance
- **Location:** `agent_code/ql`
- **Description:** QL_Agent serves as a comparative baseline, utilizing tabular Q-learning to make decisions based on a learned value function.


## How to Run the Agents

### Setting Up the Environment
1. Clone the repository:
2. Install dependencies:

### Running an Agent
To run **ppo**:

To run **QL_Agent**:

To train, add a `--train <idx-up-to-which-you-train> --n-rounds <n-rounds>`

To train without GUI, add a `--no-gui`

## Future Improvements
- 🧠 Improve **Nucleus** training to better escape bomb traps.
- 🏆 Optimize **reward structure** to incentivize proactive strategies.
- 🎯 Experiment with **alternative RL algorithms** (e.g., SAC, A2C).
- 🔄 Implement **self-play and curriculum learning** for better generalization.

---

For further details, please check the GitHub repository. 🚀

