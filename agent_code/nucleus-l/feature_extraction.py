
import torch

from agent_code.nucleus.utils import *

FEATURE_VECTOR_SIZE=29

def state_to_large_features_ppo(game_state: dict, num_coins_already_discovered: int) -> torch.tensor:
    """
    Converts the game state into a tensor feature vector for PPO.

    Board is abstracted as a boolean vector of size 29:

        If in Danger, Direction to safety:
            [0]:Up
            [1]:Down
            [2]:Left
            [3]:Right
            [4]:Wait

        Amount of danger in next moves from 0 (Safe)  to 1 (Immediately dead):
            [5]:Up
            [6]:Down
            [7]:Left
            [8]:Right
            [9]:Wait

        Direction to closest Crate (only if coins are left in game):
            [10]:Up
            [11]:Down
            [12]:Left
            [13]:Right
            [14]:Wait

        Direction in which placing a bomb would kill another player:
            [15]:Up
            [16]:Down
            [17]:Left
            [18]:Right
            [19]:Place now

        Direction to closest Coin:
            [20]:Up
            [21]:Down
            [22]:Left
            [23]:Right

        [24] Could we survive a placed Bomb?
        [25] Can we place a bomb?
        [26] Very smart bomb position (destroys 4+ crates or traps an opponent).
        [27] Normalized number of opponents left (1 = all alive, 0 = none left).
        [28] Are we currently in the lead?
    """
    feature_vector = torch.zeros(29)

    state = GameState(**game_state)
    agent_coords = state.self[3]

    # How to get to safety
    if state.is_dangerous(agent_coords):
        action_idx_to_safety = state.get_action_idx_to_closest_thing('safety')
        if action_idx_to_safety is not None:
            feature_vector[action_idx_to_safety] = 1
    elif state.is_danger_all_around(agent_coords):
        feature_vector[4] = 1  # Wait if surrounded by danger

    # How much danger is estimated in each direction
    feature_vector[5:10] = torch.tensor(state.get_danger_in_each_direction(agent_coords)).float()

    # How to get to closest crate
    if num_coins_already_discovered < NUM_COINS_IN_GAME:
        action_idx_to_crate = state.get_action_idx_to_closest_thing('crate')
        if action_idx_to_crate is not None:
            feature_vector[action_idx_to_crate + 10] = 1

    # How to get in the reach of opponents
    action_idx_to_opponents = state.get_action_idx_to_closest_thing('opponent')
    if action_idx_to_opponents is not None:
        feature_vector[action_idx_to_opponents + 15] = 1

    # How to get to closest coin
    action_idx_to_coin = state.get_action_idx_to_closest_thing('coin')
    if action_idx_to_coin is not None and action_idx_to_coin != 4:
        feature_vector[action_idx_to_coin + 20] = 1

    # Can we place a bomb and survive it?
    can_reach_safety, _ = state.simulate_own_bomb()
    feature_vector[24] = int(can_reach_safety and state.self[2])

    # Is it a perfect spot for a bomb?
    feature_vector[25] = int(state.self[2])  # Can we place a bomb?
    feature_vector[26] = int(state.is_perfect_bomb_spot(agent_coords))  # Is it a perfect bomb spot?

    # Normalized number of living opponents
    feature_vector[27] = len(state.others) / 3

    # Are we currently in the lead?
    own_score = state.self[1]
    max_opponents_score = max((opponent[1] for opponent in state.others), default=0)
    feature_vector[28] = int(own_score > max_opponents_score)

    return feature_vector