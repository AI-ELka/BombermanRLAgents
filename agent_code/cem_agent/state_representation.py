import numpy as np

def state_to_features(game_state):
    """
    Converts the game state to a feature vector.
    
    :param game_state: The game state
    :return: Feature vector representing the state
    """
    # This is a placeholder implementation
    # In a real implementation, you would extract meaningful features:
    # - Distance to nearest coin
    # - Distance to nearest crate
    # - Distance to nearest enemy
    # - Danger level (proximity to bombs)
    # - Possibility to place a bomb
    if game_state is None:
        return np.zeros(5)
    
    # Extract relevant information from the game state
    field = game_state['field']
    coins = game_state['coins']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    player_pos = game_state['self'][3]
    enemies = game_state['others']
    bombs_left = game_state['self'][2]
    
    # For now, return a simple feature vector as placeholder
    # Each feature should be normalized to a reasonable range (e.g., [0, 1])
    features = np.zeros(5)
    
    # Feature 1: Can the agent place a bomb?
    features[0] = 1.0 if bombs_left else 0.0
    
    # Feature 2: Is the agent in danger from bombs?
    in_danger = False
    for bomb_pos, bomb_countdown in bombs:
        if manhattan_distance(player_pos, bomb_pos) <= 3:
            in_danger = True
            break
    features[1] = 1.0 if in_danger else 0.0
    
    # Feature 3: Distance to nearest coin (normalized)
    if coins:
        distances = [manhattan_distance(player_pos, coin_pos) for coin_pos in coins]
        min_distance = min(distances)
        features[2] = 1.0 - min(1.0, min_distance / 10.0)  # Normalize to [0, 1]
    
    # Feature 4: Are there destroyable crates nearby?
    crates_nearby = False
    for x in range(max(0, player_pos[0] - 2), min(field.shape[0], player_pos[0] + 3)):
        for y in range(max(0, player_pos[1] - 2), min(field.shape[1], player_pos[1] + 3)):
            if field[x, y] == 1:  # 1 corresponds to crates
                crates_nearby = True
                break
    features[3] = 1.0 if crates_nearby else 0.0
    
    # Feature 5: Distance to nearest enemy (normalized)
    if enemies:
        distances = [manhattan_distance(player_pos, enemy[3]) for enemy in enemies]
        min_distance = min(distances)
        features[4] = 1.0 - min(1.0, min_distance / 15.0)  # Normalize to [0, 1]
    
    return features

def manhattan_distance(pos1, pos2):
    """
    Calculate the Manhattan distance between two points.
    
    :param pos1: Position 1
    :param pos2: Position 2
    :return: Manhattan distance
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])