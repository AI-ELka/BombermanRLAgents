from collections import namedtuple, deque
import pickle
from typing import List
import numpy as np

import events as e
from .callbacks import state_to_features, ACTIONS

# Hyperparameters
TRANSITION_HISTORY_SIZE = 1000  # Keep only the last 1000 transitions
ELITE_PERCENTAGE = 0.2  # Top 20% are considered elite
BATCH_SIZE = 50  # Number of episodes to wait before updating
N_SAMPLES = 20  # Number of parameter samples to generate

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def setup_training(self):
    """
    Initialize the training data structures:
    - self.transitions: list of all observed transitions
    - self.batch_rewards: rewards accumulated for each episode in the current batch
    - self.current_batch: counter for the current batch
    """
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.batch_rewards = []
    self.current_batch = 0
    self.episode_reward = 0
    self.train = True


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """
    Called once per step to update the agent's knowledge based on events.
    """
    self.logger.debug(f'Encountered game event(s): {", ".join(map(repr, events))}')

    # Skip if there's no old_game_state
    if old_game_state is None:
        return
    
    # Calculate reward from events
    reward = reward_from_events(self, events)
    self.episode_reward += reward
    
    # Convert states to feature representation
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)
    
    # Store the transition
    if old_features is not None and new_features is not None:
        # Convert action string to index
        action_idx = ACTIONS.index(self_action)
        self.transitions.append(Transition(old_features, action_idx, new_features, reward))


def end_of_round(self, last_game_state, last_action, events):
    """
    Called at the end of each game to update the agent's policy.
    """
    self.logger.debug(f'Encountered event(s) in final step: {", ".join(map(repr, events))}')
    
    # Calculate final reward and add to episode total
    reward = reward_from_events(self, events)
    self.episode_reward += reward
    
    # Add the last state-action transition if appropriate
    if last_game_state is not None:
        last_features = state_to_features(last_game_state)
        if last_features is not None and last_action is not None:
            action_idx = ACTIONS.index(last_action)
            # Use a dummy next state as there is none
            self.transitions.append(Transition(last_features, action_idx, None, reward))
    
    # Store this episode's total reward
    self.batch_rewards.append(self.episode_reward)
    self.episode_reward = 0
    self.current_batch += 1
    
    # Update the model when we've collected enough episodes
    if self.current_batch >= BATCH_SIZE and len(self.transitions) > 0:
        self.logger.info(f"Updating model after {self.current_batch} episodes")
        update_model_cem(self)
        self.current_batch = 0
        self.batch_rewards = []
        
    # Save the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


# Keep the existing imports and code structure, just update the reward function:

def reward_from_events(self, events: List[str]) -> float:
    """
    Calculate the reward based on game events.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5.0,           # Increased reward for collecting coins
        e.KILLED_OPPONENT: 10.0,         # Major reward for killing opponents
        e.MOVED_RIGHT: 0.05,             # Small reward for movement
        e.MOVED_LEFT: 0.05,  
        e.MOVED_UP: 0.05,    
        e.MOVED_DOWN: 0.05,
        e.WAITED: -0.1,                  # Penalty for waiting
        e.INVALID_ACTION: -1.0,          # Penalty for invalid actions
        e.BOMB_DROPPED: 0.0,             # Neutral for bombing (depends on context)
        e.CRATE_DESTROYED: 1.0,          # Good reward for destroying crates
        e.COIN_FOUND: 1.0,               # Good reward for finding coins
        e.KILLED_SELF: -10.0,            # Major penalty for killing self
        e.GOT_KILLED: -5.0               # Major penalty for getting killed
    }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    
    if e.BOMB_DROPPED in events and e.CRATE_DESTROYED not in events:
        reward_sum -= 0.5
    
    if e.BOMB_DROPPED in events and (e.KILLED_SELF in events or e.GOT_KILLED in events):
        reward_sum -= 5.0
    
    return reward_sum


def update_model_cem(self):
    """
    Update the model using the Cross-Entropy Method.
    """
    # Check if we have enough transitions
    if len(self.transitions) < 10:
        self.logger.info("Not enough transitions for meaningful update")
        return
        
    # Extract features and actions from transitions
    states = np.array([t.state for t in self.transitions if t.state is not None])
    actions = np.array([t.action for t in self.transitions if t.state is not None])
    
    # Get dimensions
    n_features = states.shape[1]
    n_actions = len(ACTIONS)
    
    # Check if model has the correct keys - if not, initialize them
    if 'weights' not in self.model:
        if 'mean' in self.model:
            # Convert old format to new format
            self.model['weights'] = np.zeros((n_features, n_actions))
            if isinstance(self.model['std'], np.ndarray) and self.model['std'].ndim == 1:
                self.model['std'] = np.ones((n_features, n_actions)) * 0.5
        else:
            self.logger.error("Model structure is incorrect")
            return
    
    # Generate parameter samples around current mean
    samples = []
    sample_returns = []
    
    for _ in range(N_SAMPLES):
        # Generate sample weights with noise based on current std
        sample_weights = self.model['weights'] + np.random.normal(0, 1, (n_features, n_actions)) * self.model['std']
        
        # Evaluate the sample by looking at what actions it would take
        total_return = 0
        for state, action, _, reward in self.transitions:
            # Get the action the sample would choose
            action_logits = np.dot(state, sample_weights)
            sample_action = np.argmax(action_logits)
            
            # Add reward only if it matches what was actually done
            if sample_action == action:
                total_return += reward
        
        samples.append(sample_weights)
        sample_returns.append(total_return)
    
    # Get indices of the elite samples
    elite_count = max(1, int(ELITE_PERCENTAGE * N_SAMPLES))
    elite_indices = np.argsort(sample_returns)[-elite_count:]
    
    # Update mean and std based on elite samples
    elite_samples = [samples[i] for i in elite_indices]
    
    # Update the mean (weights)
    self.model['weights'] = np.mean(elite_samples, axis=0)
    
    # Update the std (with a minimum to avoid collapse)
    self.model['std'] = np.std(elite_samples, axis=0) + 0.1
    
    self.logger.info(f"Updated model - avg return: {np.mean(sample_returns)}, elite return: {np.mean([sample_returns[i] for i in elite_indices])}")
    