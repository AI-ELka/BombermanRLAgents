import numpy as np

def state_to_features(game_state):
    """
    Converts the game state to a feature vector.
    
    :param game_state: The game state
    :return: Feature vector representing the state
    """
    if game_state is None:
        return np.zeros(8)  # Return an 8-dimensional feature vector
    
    # Extract relevant information from the game state
    field = game_state['field']
    coins = game_state['coins']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    player_pos = game_state['self'][3]
    enemies = game_state['others']
    bombs_left = game_state['self'][2]
    
    # Create feature vector
    features = np.zeros(8)
    
    # Feature 1: Can the agent place a bomb?
    features[0] = 1.0 if bombs_left else 0.0
    
    # Feature 2: Is the agent in danger from bombs?
    in_danger = False
    for bomb_pos, bomb_countdown in bombs:
        if manhattan_distance(player_pos, bomb_pos) <= 3:
            in_danger = True
            # More urgent if countdown is low
            features[1] = max(features[1], (4 - bomb_countdown) / 4)
    
    # Feature 3: Distance to nearest coin (normalized)
    if coins:
        distances = [manhattan_distance(player_pos, coin_pos) for coin_pos in coins]
        min_distance = min(distances)
        features[2] = 1.0 - min(1.0, min_distance / 10.0)  # Normalize to [0, 1]
        
        # Feature 4: Direction to nearest coin
        closest_coin = coins[np.argmin(distances)]
        coin_direction = np.array(closest_coin) - np.array(player_pos)
        features[3] = 1.0 if abs(coin_direction[0]) > abs(coin_direction[1]) else 0.0  # Horizontal vs vertical
    
    # Feature 5: Are there destroyable crates nearby?
    crates_nearby = False
    crate_count = 0
    for x in range(max(0, player_pos[0] - 2), min(field.shape[0], player_pos[0] + 3)):
        for y in range(max(0, player_pos[1] - 2), min(field.shape[1], player_pos[1] + 3)):
            if field[x, y] == 1:  # 1 corresponds to crates
                crates_nearby = True
                crate_count += 1
    features[4] = min(1.0, crate_count / 8.0)  # Normalize crate count
    
    # Feature 6: Is bombing effective? (would hit crates)
    bomb_would_hit_crates = False
    if bombs_left:
        for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            for distance in range(1, 4):  # Bomb range is typically 3
                x = player_pos[0] + direction[0] * distance
                y = player_pos[1] + direction[1] * distance
                if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
                    break  # Out of bounds
                if field[x, y] == -1:  # -1 is a wall
                    break  # Wall blocks explosion
                if field[x, y] == 1:  # 1 is a crate
                    bomb_would_hit_crates = True
    features[5] = 1.0 if bomb_would_hit_crates else 0.0
    
    # Feature 7: Escape routes available (to prevent suicide by bombing)
    escape_routes = 0
    for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
        next_x, next_y = player_pos[0] + dx, player_pos[1] + dy
        if (0 <= next_x < field.shape[0] and 
            0 <= next_y < field.shape[1] and 
            field[next_x, next_y] == 0):
            escape_routes += 1
    features[6] = escape_routes / 4.0  # Normalize
    
    # Feature 8: Distance to nearest enemy (normalized)
    if enemies:
        distances = [manhattan_distance(player_pos, enemy[3]) for enemy in enemies]
        min_distance = min(distances)
        features[7] = 1.0 - min(1.0, min_distance / 15.0)  # Normalize to [0, 1]
    
    return features


def manhattan_distance(pos1, pos2):
    """
    Calculate the Manhattan distance between two points.
    
    :param pos1: Position 1
    :param pos2: Position 2
    :return: Manhattan distance
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])