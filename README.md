<div align="center">

# Reinforcement Learning for Bomberman SS2024

</div>
<div align="center">
  <img src="images/bomberman.png" width="500" alt="Bomberman">
</div>

A project for the course:  
*Machine Learning Essentials*

---



## Project Overview

This project focuses on developing reinforcement learning agents for the Bomberman environment. The goal is to train an agent capable of strategic movement and decision-making to maximize its score.

### 1. Nucleus

**Framework:** Proximal Policy Optimization (PPO)  
**Average Score:** Currently being optimized  
**Location:** `agent_code/nucleus`

The Nucleus agent is the primary reinforcement learning model used in this project. While it follows a reward-based learning strategy, its performance is still being refined to improve decision-making and avoid suboptimal behaviors like staying still or making unsafe moves.

### 2. QL

**Framework:** Q-Learning  
**Average Score:** Stable performance  
**Location:** `agent_code/qlt`

The QL agent, utilizing Q-learning, provides a baseline reinforcement learning approach that performs consistently. It serves as a comparative model against Nucleus.

## Environment Overview

The Bomberman environment used in this project is a grid-based game where agents navigate a map, place bombs, and attempt to eliminate opponents while avoiding self-destruction. Key elements include:

- **Grid-Based Movement:** Agents move up, down, left, or right within a fixed game grid.
- **Bomb Mechanics:** Bombs explode after a set time, affecting nearby tiles and potentially eliminating agents.
- **Crates:** Some tiles contain crates that can be destroyed to clear paths.
- **Coins:** Collecting coins contributes to an agent’s score.
- **Opponent Interactions:** Agents must strategically avoid or target opponents for survival and higher scores.

Unlike some Bomberman versions, **this environment does not include power-ups** that enhance agent capabilities. 

---
For more details or further inquiries, please refer to the GitHub repository.

