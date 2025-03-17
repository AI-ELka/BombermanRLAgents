import numpy as np
from collections import deque
from agent_code.cem.feature_extraction import state_to_features
from agent_code.cem.cem import CEMAgent
import os

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.
    """
    # Hyperparameters
    self.MAX_COORD_HISTORY = 7
    AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
    self.MODEL_NAME = os.path.join(AGENT_DIR, "model", "cem_model.pth")
    
    # Set to True if you want to start training from scratch
    FRESH_START = False

    self.current_round = 0
    self.all_coins_game = []
    self.agent_coord_history = deque([], self.MAX_COORD_HISTORY)

    # Initialize the CEM agent
    self.agent = CEMAgent(
        logger=self.logger,
        input_size=20,  # Feature vector size
        hidden_size=128,
        output_size=6,  # Number of actions
        population_size=30,  # Match with train.py
        elite_fraction=0.3,  # Match with train.py
        device="cpu"
    )
    
    # Load model if it exists and not in training mode or if training but not a fresh start
    if (not self.train or not FRESH_START) and self.MODEL_NAME is not None:
        try:
            self.agent.load_model(self.MODEL_NAME)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            if self.train:
                self.logger.info("Starting fresh training session")

def reset_self(self, game_state: dict):
    """Reset agent state for new round."""
    self.agent_coord_history = deque([], self.MAX_COORD_HISTORY)
    self.current_round = game_state["round"]
    self.all_coins_game = []
    
    if self.train and hasattr(self, 'current_agent_idx'):
        self.agent.update_policy(episode_ended=True)

def is_new_round(self, game_state: dict) -> bool:
    return game_state["round"] != self.current_round

def act(self, game_state: dict) -> str:
    if is_new_round(self, game_state):
        reset_self(self, game_state)

    # Track agent position history
    self.agent_coord_history.append(game_state['self'][3])

    # Track all coins discovered in the game
    coins = game_state['coins']
    if coins:
        for coin in coins:
            if coin not in self.all_coins_game:
                self.all_coins_game.append(coin)
    
    num_coins_discovered = len(self.all_coins_game)
    features = state_to_features(game_state, num_coins_discovered)
    
    # In training mode, use the current agent from the population
    agent_idx = getattr(self, 'current_agent_idx', 0) if self.train else 0
    
    return self.agent.act(features, 
                        n_round=game_state["round"],
                        train=self.train,
                        agent_idx=agent_idx)