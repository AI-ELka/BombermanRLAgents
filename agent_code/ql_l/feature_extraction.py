
import torch

from agent_code.ql_l.utils import *

FEATURE_VECTOR_SIZE=24

def state_to_large_features(game_state: dict, max_opponents_score: int, num_coins_already_discovered: int) -> torch.tensor:
    """
    Board is abstracted as a boolean vector of size 29 with each feature as following:

        If in Danger, Direction to safety:
            [0]:Up
            [1]:Down
            [2]:Left
            [3]:Right
            [4]:Wait

        remove: Amount of danger in next moves from 0 (Safe)  to 1 (Immideatly dead)
            [5]:Up
            [6]:Down
            [7]:Left
            [8]:Right
            [9]:Wait

        Direction to closest Crate (only if coins are left in game)
            [10]:Up
            [11]:Down
            [12]:Left
            [13]:Right
            [14]:Wait

        Direction in which placing a bomb would kill another player
            [15]:Up
            [16]:Down
            [17]:Left
            [18]:Right
            [19]:Place now

        Direction to closest Coin
            [20]:Up
            [21]:Down
            [22]:Left
            [23]:Right

        [24] Could we survive a placed Bomb

        remove: [25] Can we place a bomb

        [26] Very smart bomb position: Would destroy 4 or more crates or opponent in trap

        [27] Normalized number of opponents left (1 = all still alive, 0 = no one left)

        [28] Currently in the lead

        Additional Ideas:
            - Number of coins left in Game (9 total)
            - sometimes placing a bomb before running out of danger can trap a opponent between two bombs
            - search for many possible paths and give back more than one action idx (especially for safety!)
                -> keep searching as long as distance is the same (Safety can search for more)
    """
    feature_vector = torch.zeros(30)

    state = GameState(**game_state)
    agent_coords = state.self[3]

    # How to get to safety
    if state.is_dangerous(agent_coords):
        action_idx_to_safety = state.get_action_idx_to_closest_thing('safety')
        if action_idx_to_safety != None:
            feature_vector[action_idx_to_safety] = 1
    elif state.is_danger_all_around(agent_coords):
        feature_vector[4] = 1

    # How much danger is estimated in each direction
    feature_vector[5:10] = torch.from_numpy(
        state.get_danger_in_each_direction(agent_coords)).type_as(feature_vector)

    # How to get to closest crate
    if num_coins_already_discovered < NUM_COINS_IN_GAME:
        action_idx_to_crate = state.get_action_idx_to_closest_thing('crate')
        if action_idx_to_crate != None:
            feature_vector[action_idx_to_crate + 10] = 1
    elif num_coins_already_discovered > NUM_COINS_IN_GAME:
        raise ValueError("Number of discovered coins cant be that high")

    # How to get in the reach of opponents
    action_idx_to_opponents = state.get_action_idx_to_closest_thing('opponent')
    if action_idx_to_opponents != None:
        feature_vector[action_idx_to_opponents + 15] = 1


    # How to get to closest coin
    action_idx_to_coin = state.get_action_idx_to_closest_thing('coin')
    if (action_idx_to_coin != None) and (action_idx_to_coin != 4):
        feature_vector[action_idx_to_coin+20] = 1

    # Can we place a Bomb and survive it?
    can_reach_safety, _ = state.simulate_own_bomb()
    feature_vector[24] = can_reach_safety and state.self[2]

    # Is it a perfect spot for a bomb?
    feature_vector[25] = state.is_perfect_bomb_spot(agent_coords)

    # Could we survive a placed Bomb
    can_reach_safety, _ = state.simulate_own_bomb()
    feature_vector[25] = can_reach_safety

    # Can we place a bomb?
    feature_vector[26] = state.self[2] 

    # Is it a perfect spot for a bomb?
    feature_vector[27] = state.is_perfect_bomb_spot(agent_coords)

    # Normalized Number of living opponent
    feature_vector[28] = len(state.others) / 3

    # Are we currently in the lead?
    own_score = state.self[1]
    feature_vector[29] = own_score > max_opponents_score

    reduced_feature_vector = torch.zeros(24)
    reduced_feature_vector[:5] = feature_vector[:5]
    reduced_feature_vector[6:21] = feature_vector[10:25]
    reduced_feature_vector[21:] = feature_vector[26:29]

    return feature_vector

