import os
import numpy as np
import pickle
from collections import deque
import events as e
from .state_representation import state_to_features
from .rewards import reward_from_events

# Actions in the Bomberman environment
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Hyperparameters
POPULATION_SIZE = 50
ELITE_FRAC = 0.2
TRANSITION_HISTORY_SIZE = 1000
LEARNING_RATE = 0.01

def setup_training(self):
    """
    Initialize training-specific data structures.
    """
    # Initialize memory for experience replay
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    
    # CEM parameters
    self.population_size = POPULATION_SIZE
    self.elite_frac = ELITE_FRAC
    self.elite_size = int(self.population_size * self.elite_frac)
    
    # Feature size depends on your state representation
    self.state_size = 5  # Adjust based on your state_to_features function
    
    if not hasattr(self, 'model') or self.model is None:
        self.model = {
            'mean': np.zeros(self.state_size),
            'std': np.ones(self.state_size),
        }

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """
    Called when a game event occurred. Use this to learn from the rewards of your actions.
    
    :param old_game_state: The state before your action
    :param self_action: The action you took
    :param new_game_state: The state after your action
    :param events: List of events that occurred
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(str, events))} in step {new_game_state["step"]}')
    
    # Skip invalid game states
    if old_game_state is None or new_game_state is None:
        return
    
    # Calculate reward from events
    reward = reward_from_events(self, events)
    
    # Store transition for later training
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)
    action_idx = ACTIONS.index(self_action) if self_action in ACTIONS else -1
    
    if action_idx != -1:
        # Add valid transition to memory
        self.transitions.append((old_features, action_idx, reward, new_features))

    # Periodically update the model using the CEM algorithm
    if len(self.transitions) >= self.population_size and new_game_state["step"] % 10 == 0:
        update_model_cem(self)

def end_of_round(self, last_game_state, last_action, events):
    """
    Called at the end of each round to update the model and save it.
    
    :param last_game_state: The final game state
    :param last_action: The last action taken
    :param events: List of events that occurred after the last action
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(str, events))} in final step')
    
    # Calculate final reward from events
    reward = reward_from_events(self, events)
    
    # Store the last transition
    if last_game_state is not None and last_action is not None:
        features = state_to_features(last_game_state)
        action_idx = ACTIONS.index(last_action) if last_action in ACTIONS else -1
        
        if action_idx != -1:
            # Terminal state (None) for new_features
            self.transitions.append((features, action_idx, reward, None))
    
    # Final update of the model
    if len(self.transitions) >= self.population_size:
        update_model_cem(self)
    
    # Save the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

def update_model_cem(self):
    """
    Update the model using the Cross-Entropy Method (CEM).
    """
    if len(self.transitions) < self.population_size:
        return
    
    # Sample transitions
    batch = np.random.choice(len(self.transitions), self.population_size, replace=False)
    states = np.array([self.transitions[i][0] for i in batch])
    
    # Generate policy variations
    policies = np.random.normal(self.model['mean'], self.model['std'], 
                                (self.population_size, len(self.model['mean'])))
    
    # Evaluate each policy
    rewards = []
    for policy in policies:
        # Compute the expected reward for this policy
        policy_reward = 0
        for i in batch:
            old_state, action, reward, _ = self.transitions[i]
            action_probs = np.dot(old_state, policy)
            predicted_action = np.argmax(action_probs)
            
            # Reward the policy if it would have chosen the same action
            if predicted_action == action:
                policy_reward += reward
        
        rewards.append(policy_reward)
    
    # Select elite policies
    rewards = np.array(rewards)
    elite_indices = np.argsort(rewards)[-self.elite_size:]
    elite_policies = policies[elite_indices]
    
    # Update the model
    self.model['mean'] = np.mean(elite_policies, axis=0)
    self.model['std'] = np.std(elite_policies, axis=0) + 1e-8  # Avoid division by zero
    
    self.logger.debug(f"Model updated: mean reward = {np.mean(rewards[elite_indices])}")