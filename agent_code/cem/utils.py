import numpy as np
from typing import List, Tuple, Dict
import queue

def is_valid_pos(field: np.ndarray, pos: tuple) -> bool:
    """Check if position is valid (within field and not a wall)."""
    x, y = pos
    return (0 <= x < field.shape[0] and 
            0 <= y < field.shape[1] and 
            field[x, y] != -1)

def is_free_pos(field: np.ndarray, pos: tuple, bombs: List[tuple], explosion_map: np.ndarray) -> bool:
    """Check if position is free (no wall, crate, bomb, or explosion)."""
    x, y = pos
    if not is_valid_pos(field, pos):
        return False
    if field[x, y] != 0:
        return False
    if explosion_map[x, y] > 0:
        return False
    for bomb_pos, _ in bombs:
        if bomb_pos == (x, y):
            return False
    return True

def manhattan_distance(pos1: tuple, pos2: tuple) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def can_destroy_crates(game_state, position):
    """
    Check if a bomb placed at position can destroy crates.
    
    Args:
        game_state: The current game state
        position: Position to check (x, y)
        
    Returns:
        bool: True if a bomb at position can destroy crates
    """
    field = game_state['field']
    x, y = position
    
    # Check all four directions from the position
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        # Check up to 3 tiles away (bomb range)
        for i in range(1, 4):
            nx, ny = x + i*dx, y + i*dy
            
            # Check if position is in bounds
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                break
                
            # If we hit a wall, explosion stops
            if field[nx, ny] == -1:
                break
                
            # If we hit a crate, bomb would destroy it
            if field[nx, ny] == 1:
                return True
    
    # No crates found in bomb's range
    return False

def find_path(start: tuple, targets: List[tuple], field: np.ndarray, 
              bombs: List[tuple], explosion_map: np.ndarray) -> List[tuple]:
    if not targets:
        return None
    
    visited = set([start])
    q = queue.Queue()
    q.put((start, []))
    
    while not q.empty():
        (x, y), path = q.get()
        
        if (x, y) in targets:
            return path
        
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:  # UP, RIGHT, DOWN, LEFT
            next_pos = (x + dx, y + dy)
            if (is_valid_pos(field, next_pos) and 
                is_free_pos(field, next_pos, bombs, explosion_map) and 
                next_pos not in visited):
                visited.add(next_pos)
                q.put((next_pos, path + [next_pos]))
    
    return None

def is_in_danger(game_state, position):
    """
    Check if a position is in danger from bombs or explosions.
    
    Args:
        game_state: The current game state
        position: Position to check (x, y)
        
    Returns:
        bool: True if the position is in danger
    """
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    field = game_state['field']
    x, y = position
    
    # Check if the position is in an active explosion
    if explosion_map[x][y] > 0:
        return True
    
    # Check for bombs about to explode
    for (bx, by), timer in bombs:
        # If timer is low, check if position is in blast range
        if timer <= 2:  # Consider bombs with timer 1 or 2 as dangerous
            # Check if on same row or column (bomb affects these directions)
            if x == bx or y == by:
                # Calculate Manhattan distance
                distance = abs(x - bx) + abs(y - by)
                
                # If within potential blast radius (3)
                if distance <= 3:
                    # Check if walls block the explosion
                    if x == bx:  # Same row
                        # Check if walls block explosion in vertical direction
                        min_y, max_y = min(y, by), max(y, by)
                        blocked = False
                        for check_y in range(min_y + 1, max_y):
                            if field[x, check_y] == -1:  # Wall blocks explosion
                                blocked = True
                                break
                        if not blocked:
                            return True
                    else:  # Same column
                        # Check if walls block explosion in horizontal direction
                        min_x, max_x = min(x, bx), max(x, bx)
                        blocked = False
                        for check_x in range(min_x + 1, max_x):
                            if field[check_x, y] == -1:  # Wall blocks explosion
                                blocked = True
                                break
                        if not blocked:
                            return True
    
    # No imminent danger detected
    return False