import numpy as np

class CEMAgent:
    def __init__(self, state_size, action_size, population_size=50, elite_fraction=0.2, noise=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.noise = noise
        self.mean = np.zeros(action_size)
        self.std = np.ones(action_size)

    def sample_actions(self):
        return np.random.normal(self.mean, self.std, (self.population_size, self.action_size))

    def evaluate_actions(self, actions, game_state):
        rewards = []
        for action in actions:
            reward = self.simulate_action(action, game_state)
            rewards.append(reward)
        return np.array(rewards)

    
    def simulate_action(self, action, game_state):
        """
        Simulate the effect of taking an action in the game environment and return the expected reward.
        
        Args:
            action: Action vector to evaluate
            game_state: Current game state dictionary
        
        Returns:
            float: Expected reward for taking the action
        """
        if game_state is None:
            return 0.0
        
        # Convert action vector to a discrete action by taking the max index
        action_idx = np.argmax(action)
        action_str = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'][action_idx]
        
        # Extract relevant game state information
        agent_pos = game_state['self'][3]
        field = game_state['field']
        bombs = game_state['bombs']
        coins = game_state['coins']
        others = [other[3] for other in game_state['others']]
        
        reward = 0.0
        
        # Check for basic movement validity
        if action_str in ['UP', 'RIGHT', 'DOWN', 'LEFT']:
            # Movement directions
            dx, dy = {
                'UP': (0, -1),
                'RIGHT': (1, 0),
                'DOWN': (0, 1),
                'LEFT': (-1, 0)
            }[action_str]
            
            new_pos = (agent_pos[0] + dx, agent_pos[1] + dy)
            
            # Check if move is invalid (wall or crate)
            if (new_pos[0] < 0 or new_pos[0] >= field.shape[0] or 
                new_pos[1] < 0 or new_pos[1] >= field.shape[1] or 
                field[new_pos] != 0):
                reward -= 5.0  # Penalty for invalid move
            else:
                reward += 0.5  # Small reward for valid movement
                
                # Reward for moving towards coins
                if coins:
                    old_nearest_coin = min([manhattan_distance(agent_pos, coin) for coin in coins])
                    new_nearest_coin = min([manhattan_distance(new_pos, coin) for coin in coins])
                    if new_nearest_coin < old_nearest_coin:
                        reward += 1.0
                    
                # Check if coin is collected
                if new_pos in coins:
                    reward += 10.0
                    
                # Reward for moving towards opponents (if we have a bomb ready)
                if game_state['self'][2] and others:  # If bomb is available
                    old_nearest_opponent = min([manhattan_distance(agent_pos, opp) for opp in others])
                    new_nearest_opponent = min([manhattan_distance(new_pos, opp) for opp in others])
                    if new_nearest_opponent < old_nearest_opponent:
                        reward += 0.5
        
        # BOMB action rewards
        elif action_str == 'BOMB':
            if not game_state['self'][2]:  # Bomb not available
                reward -= 5.0
            else:
                # Check if bomb placement is effective
                crates_nearby = 0
                opponents_nearby = 0
                
                # Count crates and opponents in bombing range
                for dx in range(-3, 4):
                    nx = agent_pos[0] + dx
                    if 0 <= nx < field.shape[0]:
                        if field[nx, agent_pos[1]] == 1:  # Crate
                            crates_nearby += 1
                        if (nx, agent_pos[1]) in others:
                            opponents_nearby += 1
                            
                for dy in range(-3, 4):
                    ny = agent_pos[1] + dy
                    if 0 <= ny < field.shape[1]:
                        if field[agent_pos[0], ny] == 1:  # Crate
                            crates_nearby += 1
                        if (agent_pos[0], ny) in others:
                            opponents_nearby += 1
                
                # Reward based on potential bombing effectiveness            
                reward += crates_nearby * 2.0
                reward += opponents_nearby * 5.0
                
                # Check escape routes after bombing
                escape_routes = 0
                for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                    nx, ny = agent_pos[0] + dx, agent_pos[1] + dy
                    if (0 <= nx < field.shape[0] and 
                        0 <= ny < field.shape[1] and 
                        field[nx, ny] == 0):
                        escape_routes += 1
                
                # Penalize bombing if there are no escape routes
                if escape_routes == 0:
                    reward -= 10.0
        
        # Check danger from bombs
        for bomb_pos, countdown in bombs:
            if ((agent_pos[0] == bomb_pos[0] and abs(agent_pos[1] - bomb_pos[1]) <= 3) or
                (agent_pos[1] == bomb_pos[1] and abs(agent_pos[0] - bomb_pos[0]) <= 3)):
                # Agent in blast radius
                danger = (4 - countdown) * 3.0  # More urgent as countdown decreases
                reward -= danger
                
                # Extra reward for moving away from bomb when in danger
                if action_str in ['UP', 'RIGHT', 'DOWN', 'LEFT']:
                    old_dist = manhattan_distance(agent_pos, bomb_pos)
                    new_dist = manhattan_distance(new_pos, bomb_pos)
                    if new_dist > old_dist:
                        reward += 3.0
        
        return reward

    def update_policy(self, actions, rewards):
        elite_count = int(self.population_size * self.elite_fraction)
        elite_indices = np.argsort(rewards)[-elite_count:]
        elite_actions = actions[elite_indices]

        self.mean = np.mean(elite_actions, axis=0)
        self.std = np.std(elite_actions, axis=0) + 1e-8  # Avoid division by zero

    def act(self, game_state):
        actions = self.sample_actions()
        rewards = self.evaluate_actions(actions, game_state)
        self.update_policy(actions, rewards)
        return self.mean  # Return the best action based on the updated policy
    

def manhattan_distance(pos1, pos2):
    """Calculate the Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])