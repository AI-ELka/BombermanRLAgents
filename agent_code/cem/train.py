import events as e
import numpy as np
import os
import atexit
from typing import List
from agent_code.cem.feature_extraction import state_to_features
from agent_code.cem.add_own_events import add_events
from agent_code.cem.own_events import *

POPULATION_SIZE = 50
ELITE_FRACTION = 0.2
EPISODES_PER_UPDATE = 1

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(AGENT_DIR, "model", "cem_model.pth")

def setup_training(self):
    """
    Initialize self for training purpose.
    This is called after `setup` in callbacks.py.
    """
    self.current_agent_idx = 0
    self.episodes_since_update = 0
    
    self.agent.population_size = POPULATION_SIZE
    self.agent.elite_frac = ELITE_FRACTION
    self.agent.generate_population()
    
    atexit.register(save_model_on_exit, self)
    self.logger.info("Registered automatic model saving at exit")

def save_model_on_exit(self):
    """Save the model when training exits."""
    try:
        # Final policy update
        self.agent.update_policy(episode_ended=True)
        
        # Save the model
        self.agent.save_model(MODEL_PATH)
        self.logger.info(f"Model automatically saved to {MODEL_PATH} on exit")
    except Exception as e:
        self.logger.error(f"Failed to save model on exit: {e}")

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
    
    Args:
        old_game_state: The state before the action
        self_action: The action that was executed
        new_game_state: The state after the action
        events: The events that occurred due to the action
    """
    if old_game_state is None:
        return
    
    # Process custom events
    events = add_events(self, old_game_state, new_game_state, events, self_action)
    
    # Calculate reward based on events
    reward = reward_from_events(self, events)
    
    # Update the reward for the current agent
    self.agent.add_reward(reward, self.current_agent_idx)
    
    # Check if this is the last step (terminal state)
    if new_game_state and new_game_state.get('terminal', False):
        # This is effectively our end_of_round handler
        handle_end_of_episode(self, new_game_state, self_action, events)
    
    # Store information for debugging
    if new_game_state is not None:
        self.logger.debug(f"Agent {self.current_agent_idx}, Round {new_game_state['round']}, Reward: {reward}")

def handle_end_of_episode(self, game_state, last_action, events):
    """Handle the end of an episode even if end_of_round isn't called."""
    self.logger.info("Episode finished, handling end of round")
    
    # Process final events with custom events
    try:
        events = add_events(self, game_state, None, events, last_action)
    except Exception as e:
        self.logger.error(f"Error adding events: {e}")
    
    # Calculate final reward
    reward = reward_from_events(self, events)
    
    # Add bonus for surviving longer rounds
    if hasattr(game_state, 'round'):
        round_bonus = min(game_state['round'] * 0.5, 10)  # Cap at 10
        reward += round_bonus
    
    # Add final reward to the current agent
    self.agent.add_reward(reward, self.current_agent_idx)
    
    # Log episode performance
    self.logger.info(f"Episode ended. Agent {self.current_agent_idx}, " +
                    f"Round: {game_state.get('round', 0)}, " +
                    f"Final reward: {reward:.2f}")
    
    # Handle episode completion
    self.episodes_since_update += 1
    
    if self.episodes_since_update >= EPISODES_PER_UPDATE:
        # Time to move to the next agent in population
        self.current_agent_idx = (self.current_agent_idx + 1) % POPULATION_SIZE
        self.episodes_since_update = 0
        
        # If we've gone through all agents, update policy
        if self.current_agent_idx == 0:
            # Update the policy based on collected rewards
            policy_updated = self.agent.update_policy(episode_ended=True)
            
            if policy_updated:
                # Save model after each policy update
                try:
                    self.agent.save_model(MODEL_PATH)
                    self.logger.info(f"Model saved to {MODEL_PATH}")
                except Exception as e:
                    self.logger.error(f"Error saving model: {e}")
    
    # Reset visited positions for the next episode
    if hasattr(self, 'visited_positions'):
        self.visited_positions = set()


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game to evaluate final rewards and outcomes.
    """
    self.logger.info("Original end_of_round called!")
    
    # Process final events
    events = add_events(self, last_game_state, None, events, last_action)
    reward = reward_from_events(self, events)
    self.agent.add_reward(reward, self.current_agent_idx)
    
    self.episodes_since_update += 1
    
    if self.episodes_since_update >= EPISODES_PER_UPDATE:
        self.current_agent_idx = (self.current_agent_idx + 1) % POPULATION_SIZE
        self.episodes_since_update = 0
        
        if self.current_agent_idx == 0:
            self.agent.update_policy(episode_ended=True)
            self.agent.save_model(MODEL_PATH)
        
            mean_reward = np.mean(self.agent.rewards)
            self.logger.info(f"Policy updated. Mean reward: {mean_reward:.4f}")

def reward_from_events(self, events: List[str]) -> float:
    """
    Calculate the reward based on event list with stronger movement incentives.
    """
    reward_map = {
        e.MOVED_LEFT: 0.01,  # Small positive reward for any movement
        e.MOVED_RIGHT: 0.01,
        e.MOVED_UP: 0.01,
        e.MOVED_DOWN: 0.01,
        e.WAITED: -0.3,  # Stronger penalty for waiting
        e.INVALID_ACTION: -1.5,  # Stronger penalty for invalid actions
        e.BOMB_DROPPED: -0.1,  # Reduced penalty for bombs
        
        e.CRATE_DESTROYED: 1.0,  # Higher reward for progress
        e.COIN_FOUND: 2.0,
        e.COIN_COLLECTED: 10.0,
        
        e.KILLED_OPPONENT: 25.0,
        e.KILLED_SELF: -15.0,
        
        e.GOT_KILLED: -5.0,
        e.SURVIVED_ROUND: 10.0,  # Higher survival reward
        
        MOVED_TOWARDS_COIN: 0.5,  # Stronger directional incentives
        MOVED_AWAY_FROM_COIN: -0.5,
        MOVED_TOWARDS_CRATE: 0.3,
        MOVED_AWAY_FROM_CRATE: -0.3,
        MOVED_TO_SAFETY: 1.0,  # Strong safety incentive
        STAYED_IN_DANGER: -2.0,  # Strong penalty for staying in danger
        BOMB_PLACED_NEAR_ENEMY: 2.0,
        USEFUL_BOMB: 1.0,
        USELESS_BOMB: -1.0
    }
    
    # Calculate base reward
    reward = sum(reward_map.get(event, 0) for event in events)
    
    # Add a small random noise to break symmetry in similar states
    reward += np.random.normal(0, 0.01)
    
    return reward