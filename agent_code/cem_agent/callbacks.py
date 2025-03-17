import os
import numpy as np
import pickle
from .state_representation import state_to_features

# Actions in the Bomberman environment
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.
    """
    if os.path.isfile("my-saved-model.pt"):
        self.logger.info("Loading model from file...")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    else:
        self.logger.info("Setting up new CEM model...")
        # Initialize a new CEM model
        state_size = 5  # Adjust based on your feature representation
        action_size = len(ACTIONS)
        self.model = {
            'mean': np.zeros(state_size),
            'std': np.ones(state_size),
        }

def act(self, game_state):
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param game_state: The game state
    :return: The selected action
    """
    # Extract features from game state
    features = state_to_features(game_state)
    
    # In a real CEM implementation, you'd use the learned policy to select an action
    # For now, let's implement a simple approach using the learned mean vector
    if hasattr(self, 'model') and 'mean' in self.model:
        # Use the learned mean vector to make a decision
        # This is a simplified version of action selection
        action_probs = np.dot(features, self.model['mean'])
        action_idx = np.argmax(action_probs)
        print(ACTIONS[action_idx % len(ACTIONS)])
        return ACTIONS[action_idx % len(ACTIONS)]
    else:
        # Fallback to random action
        self.logger.debug("No model available, choosing action at random")
        return np.random.choice(ACTIONS)