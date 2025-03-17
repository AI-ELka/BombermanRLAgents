import os
import pickle
import numpy as np
from random import shuffle
from .state_representation import state_to_features, manhattan_distance  # Import from state_representation module

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
            # Initialize weights with small random values instead of zeros
            self.model = {
                'weights': np.random.uniform(-0.1, 0.1, (feature_size, len(ACTIONS))),
                'std': np.ones((feature_size, len(ACTIONS))) * 0.5,
            }
            
            # Initialize bomb action with negative weight to discourage bombing at start
            bomb_index = ACTIONS.index('BOMB')
            self.model['weights'][:, bomb_index] = -0.2
            
        else:
            self.logger.error("Failed to determine feature size.")
            self.model = None
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state):
    """
    Your agent should parse the input, think, and take a decision.
    Apply game logic here.
    
    :param self: The same object that is passed to all callbacks.
    :param game_state: The dictionary describing the current game board.
    :return: The action to take as a string.
    """
    # Extract features from game state
    features = state_to_features(game_state)
    
    # Add simple safety check to prevent immediate suicide
    if hasattr(self, 'model') and 'weights' in self.model and features is not None:
        # Compute action probabilities using the policy weights
        action_logits = np.dot(features, self.model['weights'])
        
        # Safety mechanism: Check if placing a bomb is dangerous
        bomb_action = ACTIONS.index('BOMB')
        if is_bomb_dangerous(game_state):
            action_logits[bomb_action] = float('-inf')  # Make bombing impossible
        
        # Choose the action with the highest score
        action_idx = np.argmax(action_logits)
        return ACTIONS[action_idx]
    else:
        # Fallback to random action
        self.logger.debug("No model available or invalid state, choosing action randomly")
        return np.random.choice([a for a in ACTIONS if a != 'BOMB'])  # Avoid bombs initially

def is_bomb_dangerous(game_state):
    """Check if placing a bomb would lead to agent's death."""
    if game_state is None:
        return True
        
    agent_pos = game_state['self'][3]
    field = game_state['field']
    
    # Check if agent is trapped (no escape routes)
    escape_routes = 0
    for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
        next_x, next_y = agent_pos[0] + dx, agent_pos[1] + dy
        # Check if the position is within bounds and walkable
        if (0 <= next_x < field.shape[0] and 
            0 <= next_y < field.shape[1] and 
            field[next_x, next_y] == 0):
            escape_routes += 1
    
    # If there are fewer than 2 escape routes, bombing is dangerous
    return escape_routes < 2
    

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