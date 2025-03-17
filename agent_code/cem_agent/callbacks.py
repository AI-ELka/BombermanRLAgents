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
            # Initialize weights with small random values instead of zeros
            self.model = {
                'weights': np.random.uniform(-0.1, 0.1, (feature_size, len(ACTIONS))),
                'std': np.ones((feature_size, len(ACTIONS))) * 0.5,
            }
            
            # Initialize bomb action with large negative weight to strongly discourage bombing at start
            bomb_index = ACTIONS.index('BOMB')
            self.model['weights'][:, bomb_index] = -2.0  # Very negative to prevent bombing early
            
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
    
    # Special case: If the agent is in immediate danger from a bomb, prioritize escape
    if game_state is not None and is_in_danger(game_state):
        # Find safe escape direction
        safe_move = find_safe_move(game_state)
        if safe_move:
            self.logger.debug(f"DANGER! Moving to safety: {safe_move}")
            return safe_move
    
    # Regular action selection
    if hasattr(self, 'model') and 'weights' in self.model and features is not None:
        # Compute action logits using the policy weights
        action_logits = np.dot(features, self.model['weights'])
        
        # Safety mechanism: Check if placing a bomb is dangerous
        bomb_action = ACTIONS.index('BOMB')
        if is_bomb_dangerous(game_state):
            action_logits[bomb_action] = float('-inf')  # Make bombing impossible
        
        # Debug info
        self.logger.debug(f"Action logits: {action_logits}")
        
        # Choose the action with the highest score
        action_idx = np.argmax(action_logits)
        return ACTIONS[action_idx]
    else:
        # Fallback to random movement (no bombing)
        self.logger.debug("No model available or invalid state, choosing action randomly")
        return np.random.choice([a for a in ACTIONS if a != 'BOMB'])  # Avoid bombs initially


def is_in_danger(game_state):
    """Check if agent is in immediate danger from bombs."""
    if game_state is None:
        return False
        
    agent_pos = game_state['self'][3]
    bombs = game_state['bombs']
    
    # Check if any bomb could hit the agent
    for bomb_pos, countdown in bombs:
        if manhattan_distance(agent_pos, bomb_pos) <= 3:  # Within blast radius
            return True  # Agent is in danger
    
    return False


def find_safe_move(game_state):
    """Find a safe move away from bombs."""
    if game_state is None:
        return None
        
    agent_pos = game_state['self'][3]
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    
    # Evaluate safety of each possible move
    moves = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, right, down, left
    
    # Calculate safety scores for each direction
    safety_scores = []
    
    for i, (dx, dy) in enumerate(directions):
        nx, ny = agent_pos[0] + dx, agent_pos[1] + dy
        
        # Check if move is valid
        if nx < 0 or nx >= field.shape[0] or ny < 0 or ny >= field.shape[1] or field[nx, ny] != 0:
            safety_scores.append(-999)  # Invalid move
            continue
            
        # Start with a base safety score
        safety = 0
        
        # Decrease safety if the position is within explosion radius of a bomb
        for bomb_pos, countdown in bombs:
            bomb_x, bomb_y = bomb_pos
            
            # If in same row or column as bomb and within bomb blast radius (3)
            if ((nx == bomb_x and abs(ny - bomb_y) <= 3) or
                (ny == bomb_y and abs(nx - bomb_x) <= 3)):
                safety -= (4 - countdown) * 10  # Urgency increases as countdown decreases
                
        # Decrease safety if on explosion map
        if explosion_map[nx, ny] > 0:
            safety -= 100
            
        # Lower safety for tiles closer to bombs
        for bomb_pos, _ in bombs:
            distance = manhattan_distance((nx, ny), bomb_pos)
            safety -= max(0, 5 - distance)  # Penalize positions close to bombs
            
        safety_scores.append(safety)
    
    # Find the safest move that's valid
    if max(safety_scores) > -999:
        best_move_idx = np.argmax(safety_scores)
        return moves[best_move_idx]
    
    # If no safe move, try WAIT (as a last resort)
    return 'WAIT'


def is_bomb_dangerous(game_state):
    """Check if placing a bomb would lead to agent's death."""
    if game_state is None:
        return True
        
    agent_pos = game_state['self'][3]
    field = game_state['field']
    
    # Check for existing bombs nearby
    for bomb_pos, countdown in game_state['bombs']:
        if manhattan_distance(agent_pos, bomb_pos) <= 3:
            return True  # Don't place bombs near other bombs
    
    # Count escape routes and check if any are truly safe
    escape_routes = 0
    safe_escape_routes = 0
    
    for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
        next_x, next_y = agent_pos[0] + dx, agent_pos[1] + dy
        
        # Check if position is walkable
        if (0 <= next_x < field.shape[0] and 
            0 <= next_y < field.shape[1] and 
            field[next_x, next_y] == 0):
            escape_routes += 1
            
            # Check if this escape route leads to more escape routes (truly safe)
            further_escapes = 0
            for fx, fy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                third_x, third_y = next_x + fx, next_y + fy
                # Exclude the original position
                if (third_x, third_y) != agent_pos:
                    if (0 <= third_x < field.shape[0] and 
                        0 <= third_y < field.shape[1] and 
                        field[third_x, third_y] == 0):
                        further_escapes += 1
            
            if further_escapes >= 1:
                safe_escape_routes += 1
    
    # Are there useful targets for bombing?
    bomb_is_useful = False
    
    # Check for crates or enemies in bomb range
    for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        for distance in range(1, 4):  # Bomb range is typically 3
            x = agent_pos[0] + direction[0] * distance
            y = agent_pos[1] + direction[1] * distance
            
            # Check bounds
            if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
                break
                
            # Stop at walls
            if field[x, y] == -1:
                break
                
            # Found a crate (useful for bombing)
            if field[x, y] == 1:
                bomb_is_useful = True
    
    # Check for enemies
    for enemy in game_state['others']:
        enemy_pos = enemy[3]
        if manhattan_distance(agent_pos, enemy_pos) <= 3:
            # Check if enemy is in line with agent (can be hit by bomb)
            if enemy_pos[0] == agent_pos[0] or enemy_pos[1] == agent_pos[1]:
                # Check for walls between agent and enemy
                can_hit = True
                if enemy_pos[0] == agent_pos[0]:  # same column
                    for y in range(min(agent_pos[1], enemy_pos[1]) + 1, max(agent_pos[1], enemy_pos[1])):
                        if field[agent_pos[0], y] != 0:
                            can_hit = False
                            break
                else:  # same row
                    for x in range(min(agent_pos[0], enemy_pos[0]) + 1, max(agent_pos[0], enemy_pos[0])):
                        if field[x, agent_pos[1]] != 0:
                            can_hit = False
                            break
                
                if can_hit:
                    bomb_is_useful = True
    
    # Only allow bombing if:
    # 1. There are at least 2 escape routes
    # 2. At least one escape route leads to further safe positions
    # 3. Either there are crates to destroy or enemies to hit
    return not (escape_routes >= 2 and safe_escape_routes >= 1 and bomb_is_useful)


def manhattan_distance(pos1, pos2):
    """
    Calculate the Manhattan distance between two points.
    
    :param pos1: Position 1
    :param pos2: Position 2
    :return: Manhattan distance
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def state_to_features(game_state):
    """
    Converts the game state to a feature vector.
    
    :param game_state: A dictionary describing the current game board.
    :return: A numpy array of features.
    """
    # If there is no state, return zeros (for initialization)
    if game_state is None:
        return np.zeros(10)
    
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
        nearest_coin = min([manhattan_distance(agent_position, coin_pos) 
                            for coin_pos in game_state['coins']])
        features[4] = 1 / (1 + nearest_coin)  # Closer coins have higher values
    
    # 6. Can we place a bomb?
    features[5] = 1.0 if game_state['self'][2] else 0.0
    
    # 7. Are there crates nearby to bomb?
    crates_nearby = 0
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if abs(dx) + abs(dy) <= 3:  # Manhattan distance <= 3
                nx, ny = x + dx, y + dy
                if (0 <= nx < field.shape[0] and 
                    0 <= ny < field.shape[1] and 
                    field[nx, ny] == 1):  # 1 = crate
                    crates_nearby += 1
    features[6] = min(1.0, crates_nearby / 8)  # Normalize
    
    # 8. Is the agent in danger from bombs?
    in_danger = 0
    for bomb_pos, countdown in game_state['bombs']:
        if manhattan_distance(agent_position, bomb_pos) <= 3:
            in_danger = max(in_danger, (4 - countdown) / 4)  # Higher value = more danger
    features[7] = in_danger
    
    # 9. Escape routes available
    escape_routes = 0
    for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
        next_x, next_y = x + dx, y + dy
        if (0 <= next_x < field.shape[0] and 
            0 <= next_y < field.shape[1] and 
            field[next_x, next_y] == 0):
            escape_routes += 1
    features[8] = escape_routes / 4  # Normalize
    
    # 10. Are there opponents nearby?
    if game_state['others']:
        nearest_opponent = min([manhattan_distance(agent_position, other[3]) 
                               for other in game_state['others']])
        features[9] = 1 / (1 + nearest_opponent)  # Closer opponents have higher values
    
    return features