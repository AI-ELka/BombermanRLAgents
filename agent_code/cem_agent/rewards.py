import events as e

def reward_from_events(self, events):
    """
    Calculates the reward for a given set of events.
    
    :param events: List of events
    :return: Reward value
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -100,
        e.CRATE_DESTROYED: 5,
        e.BOMB_DROPPED: 1,
        e.INVALID_ACTION: -5,
        e.WAITED: -1,
        e.MOVED_LEFT: -0.1,
        e.MOVED_RIGHT: -0.1,
        e.MOVED_UP: -0.1,
        e.MOVED_DOWN: -0.1
    }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
            
    self.logger.info(f"Reward for events {events}: {reward_sum}")
    return reward_sum