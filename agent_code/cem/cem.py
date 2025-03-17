import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """Neural network policy for Cross Entropy Method."""
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Initialize with slightly larger values to encourage exploration
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # No activation here, will apply softmax later

class CEMAgent:
    """Cross Entropy Method agent for Bomberman."""
    def __init__(self, logger, input_size=20, hidden_size=128, output_size=6, 
                 population_size=50, elite_fraction=0.2, device='cpu'):
        self.logger = logger
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.population_size = population_size
        self.elite_frac = elite_fraction
        self.n_elite = max(2, int(population_size * elite_fraction))  # At least 2 elites
        self.device = torch.device(device)
        
        # Action mapping
        self.actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
        
        # Initialize policy network
        self.policy = PolicyNetwork(input_size, hidden_size, output_size).to(self.device)
        
        # Initialize population parameters
        self.population = []
        self.rewards = np.zeros(population_size)
        
        # For exploration during training
        self.temperature = 3.0  # Start with higher temperature for more exploration
        self.min_temperature = 0.5  # Don't go too low
        self.temp_decay = 0.995
        
        # Track training metrics
        self.episode_count = 0
        self.best_reward = float('-inf')
        self.no_improvement_count = 0
        
        # Generate initial population with higher variance
        self.param_std = None
        self.initial_std = 0.5  # Higher initial variance
    
    def generate_population(self):
        """Generate a new population of policy parameters with more variance."""
        self.population = []
        
        # Get reference to all parameters
        params = list(self.policy.parameters())
        param_shapes = [p.data.shape for p in params]
        
        # Current mean parameters
        mean_params = [p.data.clone().cpu().numpy() for p in params]
        
        # Initialize std if not yet defined
        if self.param_std is None:
            self.param_std = [np.ones_like(p) * self.initial_std for p in mean_params]
        
        # Generate population
        for _ in range(self.population_size):
            new_params = []
            for i, (mean, std, shape) in enumerate(zip(mean_params, self.param_std, param_shapes)):
                new_param = mean + np.random.normal(0, std, size=shape)
                new_params.append(new_param)
            self.population.append(new_params)
        
        # Reset rewards
        self.rewards = np.zeros(self.population_size)
    
    def update_policy(self, episode_ended=False):
        """Update policy based on collected rewards."""
        if not self.population or not episode_ended:
            return False  # No update performed
        
        self.episode_count += 1
        
        # Check if we should update
        if not self.rewards.any():  # If all rewards are zero
            self.logger.info("All rewards are zero, skipping update")
            return False
            
        # Sort population by reward
        sorted_indices = np.argsort(self.rewards)[::-1]  # Descending order
        elite_indices = sorted_indices[:self.n_elite]
        
        # Extract elite parameters
        elite_params = [self.population[i] for i in elite_indices]
        
        # Compute new mean and std from elite population
        new_means = []
        new_stds = []
        
        for i in range(len(elite_params[0])):
            param_stack = np.stack([params[i] for params in elite_params])
            new_mean = np.mean(param_stack, axis=0)
            new_std = np.std(param_stack, axis=0)
            
            # Add minimum std to prevent premature convergence
            new_std = np.maximum(new_std, 0.05)  # Higher minimum std
            
            new_means.append(new_mean)
            new_stds.append(new_std)
        
        # Track best reward for early stopping
        current_best = np.max(self.rewards)
        if current_best > self.best_reward:
            self.best_reward = current_best
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            # If no improvement for a while, increase exploration
            if self.no_improvement_count >= 5:
                self.logger.info(f"No improvement for {self.no_improvement_count} updates, increasing exploration")
                new_stds = [std * 1.5 for std in new_stds]  # Increase exploration
                self.temperature = min(3.0, self.temperature * 1.2)  # Increase temperature
                self.no_improvement_count = 0
        
        # Update policy with new mean parameters
        params = list(self.policy.parameters())
        for i, p in enumerate(params):
            p.data = torch.tensor(new_means[i], device=self.device).float()
        
        # Update std for next generation
        self.param_std = new_stds
        
        # Log statistics
        self.logger.info(f"Update {self.episode_count}: Best reward: {current_best:.2f}, " +
                        f"Mean reward: {np.mean(self.rewards):.2f}, " +
                        f"Temperature: {self.temperature:.2f}")
        
        # Reset rewards 
        self.rewards = np.zeros(self.population_size)
        
        # Decay exploration temperature (slower decay)
        self.temperature = max(self.min_temperature, self.temperature * self.temp_decay)
        
        # Generate new population
        self.generate_population()
        self.current_agent_idx = 0
        
        return True  # Update was performed
    
    def apply_agent_params(self, idx):
        """Apply parameters of agent at index idx to the policy."""
        if not self.population:
            self.generate_population()
            self.current_agent_idx = 0
        
        idx = idx % self.population_size
        agent_params = self.population[idx]
        
        params = list(self.policy.parameters())
        for i, p in enumerate(params):
            p.data = torch.tensor(agent_params[i], device=self.device).float()
    
    def add_reward(self, reward, agent_idx):
        """Add reward for agent at given index."""
        if agent_idx < len(self.rewards):
            self.rewards[agent_idx] += reward
    
    def act(self, features, n_round=0, train=False, agent_idx=0):
        """
        Choose an action based on the feature vector.
        
        Args:
            features: Feature vector
            n_round: Current game round
            train: Whether in training mode
            agent_idx: Index of the agent in the population (for training)
        
        Returns:
            Selected action as a string
        """
        if features is None:
            # Choose a random action if no features are available
            return np.random.choice(self.actions)
        
        if train:
            # Apply parameters of the specified agent
            self.apply_agent_params(agent_idx)
        
        # Convert features to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        # Forward pass through the policy network
        with torch.no_grad():
            logits = self.policy(features_tensor)
            
            # In training, add temperature for exploration
            if train:
                logits = logits / self.temperature
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=0).cpu().numpy()
        
        # Ensure we don't have too many WAIT actions by reducing its probability
        if train:
            # Reduce probability of WAIT and BOMB actions to encourage movement
            probs[4] *= 0.5  # WAIT
            probs[5] *= 0.8  # BOMB
            probs = probs / probs.sum()  # Re-normalize
            
            action_idx = np.random.choice(len(self.actions), p=probs)
        else:
            # In evaluation, don't always choose the best action (slight exploration)
            if np.random.random() < 0.1:  # 10% exploration in evaluation
                # Sample from a softmax with lower temperature
                temp_probs = F.softmax(logits / 0.5, dim=0).cpu().numpy()
                action_idx = np.random.choice(len(self.actions), p=temp_probs)
            else:
                action_idx = np.argmax(probs)
        
        return self.actions[action_idx]
    
    def save_model(self, filename):
        """Save the policy model to a file."""
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'temperature': self.temperature,
            'best_reward': self.best_reward
        }, filename)
        self.logger.info(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load the policy model from a file."""
        try:
            checkpoint = torch.load(filename, map_location=self.device)
            
            # Create a new policy network if architecture doesn't match
            if (self.input_size != checkpoint['input_size'] or 
                self.hidden_size != checkpoint['hidden_size'] or
                self.output_size != checkpoint['output_size']):
                self.input_size = checkpoint['input_size']
                self.hidden_size = checkpoint['hidden_size']
                self.output_size = checkpoint['output_size']
                self.policy = PolicyNetwork(self.input_size, self.hidden_size, self.output_size).to(self.device)
            
            # Load model parameters
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.temperature = checkpoint.get('temperature', self.min_temperature)
            self.best_reward = checkpoint.get('best_reward', float('-inf'))
            self.logger.info(f"Model loaded from {filename}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")