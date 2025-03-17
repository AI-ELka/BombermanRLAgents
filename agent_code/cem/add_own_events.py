import numpy as np
import events as e
from agent_code.cem.own_events import *
from agent_code.cem.utils import *
from agent_code.cem.feature_extraction import state_to_features

def add_events(self, old_state, new_state, events, action):
    """
    Adds custom events based on the old and new game states.
    """
    if old_state is None:
        return events
    
    old_pos = old_state['self'][3]
    
    # Check if any movement happened
    if action in ['UP', 'RIGHT', 'DOWN', 'LEFT'] and new_state is not None:
        new_pos = new_state['self'][3]
        
        # Detect if agent actually moved or was blocked
        if old_pos == new_pos and action != 'WAIT':
            events.append(AGENT_BLOCKED)
    
    # If new state is None (end of round), we can't do further comparisons
    if new_state is None:
        return events
    
    new_pos = new_state['self'][3]
    
    # Add events for movement towards/away from coins
    if action in ['UP', 'RIGHT', 'DOWN', 'LEFT']:
        try:
            old_features = state_to_features(old_state, len(self.all_coins_game))
            new_features = state_to_features(new_state, len(self.all_coins_game))
            
            # Check if moving towards coin
            if any(old_features[15:19]):
                # Determine if we moved in the direction of the coin
                coin_direction_mapping = {
                    'UP': 15,     # Up feature index
                    'DOWN': 16,   # Down feature index
                    'LEFT': 17,   # Left feature index
                    'RIGHT': 18   # Right feature index
                }
                
                if old_features[coin_direction_mapping.get(action, -1)] == 1:
                    events.append(MOVED_TOWARDS_COIN)
                else:
                    events.append(MOVED_AWAY_FROM_COIN)
            
            # Check if moving towards crate
            if any(old_features[5:9]):
                # Determine if we moved in the direction of the crate
                crate_direction_mapping = {
                    'UP': 5,      # Up feature index
                    'DOWN': 6,    # Down feature index
                    'LEFT': 7,    # Left feature index
                    'RIGHT': 8    # Right feature index
                }
                
                if old_features[crate_direction_mapping.get(action, -1)] == 1:
                    events.append(MOVED_TOWARDS_CRATE)
                else:
                    events.append(MOVED_AWAY_FROM_CRATE)
        except Exception as e:
            self.logger.error(f"Error in feature extraction: {e}")
    
    # Check for safety-related events
    if is_in_danger(old_state, old_pos):
        if not is_in_danger(new_state, new_pos):
            events.append(MOVED_TO_SAFETY)
        else:
            events.append(STAYED_IN_DANGER)
    
    # Bomb placement events
    if action == 'BOMB':
        # Check if bomb is near opponent
        opponents = [opponent[3] for opponent in old_state.get('others', [])]
        if any(manhattan_distance(old_pos, opp) <= 3 for opp in opponents):
            events.append(BOMB_PLACED_NEAR_ENEMY)
        
        # Check if bomb can destroy crates
        if can_destroy_crates(old_state, old_pos):
            events.append(USEFUL_BOMB)
        else:
            events.append(USELESS_BOMB)
    
    # Add exploration incentive - reward visiting new positions
    if not hasattr(self, 'visited_positions'):
        self.visited_positions = set()
    
    pos_tuple = tuple(new_pos)
    if pos_tuple not in self.visited_positions:
        self.visited_positions.add(pos_tuple)
        events.append(EXPLORED_NEW_TILE)
    
    return events