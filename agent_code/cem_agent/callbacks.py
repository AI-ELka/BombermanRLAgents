import os
import pickle
import numpy as np
from random import shuffle

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    """
    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        # Initialize model with proper dimensions based on feature space
        features = state_to_features(None)  # Get feature dimension
        if features is not None:
            feature_size = len(features)
            # For each action, we'll have weights to compute policy
            self.model = {
                'weights': np.zeros((feature_size, len(ACTIONS))),  # Changed from 'mean' to 'weights'
                'std': np.ones((feature_size, len(ACTIONS))) * 0.5,  # Matrix, not vector
            }
        else:
            self.logger.error("Failed to determine feature size.")
            self.model = None
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state):
    # Extract features from game state
    features = state_to_features(game_state)
    
    # Use the right key 'weights' instead of 'mean'
    if hasattr(self, 'model') and 'weights' in self.model and features is not None:
        # Compute action probabilities using the policy weights
        action_logits = np.dot(features, self.model['weights'])
        
        # Choose the action with the highest score
        action_idx = np.argmax(action_logits)
        return ACTIONS[action_idx]
    else:
        # Fallback to random action
        self.logger.debug("No model available or invalid state, choosing action randomly")
        return np.random.choice(ACTIONS)
    

def state_to_features(game_state):
    """
    Converts the game state to a feature vector.
    
    :param game_state: A dictionary describing the current game board.
    :return: A numpy array of features.
    """
    # If there is no state, return None
    if game_state is None:
        return None
    
    # Get the position of our agent
    agent_position = game_state['self'][3]
    
    # Initialize the feature vector
    features = np.zeros(10)  # Adjust the size according to your feature design

    # Example features (you can expand these):
    # 1-4. Distance to walls in each direction (up, right, down, left)
    field = game_state['field']
    
    # Simple features: distance to walls in each direction
    x, y = agent_position
    
    # Distance to wall/obstacle up
    wall_up = 0
    for i in range(1, 17):  # Max field size
        if y - i < 0 or field[x, y - i] != 0:
            wall_up = i
            break
    features[0] = wall_up / 17  # Normalize
    
    # Distance to wall/obstacle right
    wall_right = 0
    for i in range(1, 17):
        if x + i >= field.shape[0] or field[x + i, y] != 0:
            wall_right = i
            break
    features[1] = wall_right / 17
    
    # Distance to wall/obstacle down
    wall_down = 0
    for i in range(1, 17):
        if y + i >= field.shape[1] or field[x, y + i] != 0:
            wall_down = i
            break
    features[2] = wall_down / 17
    
    # Distance to wall/obstacle left
    wall_left = 0
    for i in range(1, 17):
        if x - i < 0 or field[x - i, y] != 0:
            wall_left = i
            break
    features[3] = wall_left / 17
    
    # 5. Is there a coin nearby?
    if game_state['coins']:
        nearest_coin = min([np.linalg.norm(np.array(agent_position) - np.array(coin_pos)) 
                            for coin_pos in game_state['coins']])
        features[4] = 1 / (1 + nearest_coin)  # Closer coins have higher values
    
    # 6. Can we place a bomb?
    features[5] = 1.0 if game_state['self'][2] else 0.0
    
    # 7-10. Are there opponents nearby?
    if game_state['others']:
        nearest_opponent = min([np.linalg.norm(np.array(agent_position) - np.array(other[3])) 
                               for other in game_state['others']])
        features[6] = 1 / (1 + nearest_opponent)  # Closer opponents have higher values
    
    # Other features could include:
    # - Direction to nearest coin
    # - Direction to nearest crate
    # - Is the agent in danger from a bomb?

    return features