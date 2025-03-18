import numpy as np

class CEMAgent:
    def __init__(self, state_size, action_size, population_size=50, elite_fraction=0.2, noise=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.noise = noise
        self.mean = np.zeros(action_size)
        self.std = np.ones(action_size)

    def sample_actions(self):
        return np.random.normal(self.mean, self.std, (self.population_size, self.action_size))

    def evaluate_actions(self, actions, game_state):
        rewards = []
        for action in actions:
            reward = self.simulate_action(action, game_state)
            rewards.append(reward)
        return np.array(rewards)

    def simulate_action(self, action, game_state):
        return np.random.rand()

    def update_policy(self, actions, rewards):
        elite_count = int(self.population_size * self.elite_fraction)
        elite_indices = np.argsort(rewards)[-elite_count:]
        elite_actions = actions[elite_indices]

        self.mean = np.mean(elite_actions, axis=0)
        self.std = np.std(elite_actions, axis=0) + 1e-8  # Avoid division by zero

    def act(self, game_state):
        actions = self.sample_actions()
        rewards = self.evaluate_actions(actions, game_state)
        self.update_policy(actions, rewards)
        return self.mean  # Return the best action based on the updated policy
