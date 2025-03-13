import os
import random
from typing import Callable

import events as e
import agent_code.dql.own_events as own_e
from agent_code.dql.utils import *

import torch

class QNetwork(torch.nn.Module):
  """
  A Q-Network implemented with PyTorch.

  Attributes
  ----------
  layer1 : torch.nn.Linear
      First fully connected layer.
  layer2 : torch.nn.Linear
      Second fully connected layer.
  layer3 : torch.nn.Linear
      Third fully connected layer.

  Methods
  -------
  forward(x: torch.Tensor) -> torch.Tensor
      Define the forward pass of the QNetwork.
  """

  def __init__(self,
               n_observations: int,
               n_actions: int,
               nn_l1: int,
               nn_l2: int):
    """
    Initialize a new instance of QNetwork.

    Parameters
    ----------
    n_observations : int
        The size of the observation space.
    n_actions : int
        The size of the action space.
    nn_l1 : int
        The number of neurons on the first layer.
    nn_l2 : int
        The number of neurons on the second layer.
    """
    super(QNetwork, self).__init__()

    self.layer1 = torch.nn.Linear(n_observations, nn_l1)
    self.layer2 = torch.nn.Linear(nn_l1, nn_l2)
    self.layer3 = torch.nn.Linear(nn_l2, n_actions)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Define the forward pass of the QNetwork.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor (state).

    Returns
    -------
    torch.Tensor
        The output tensor (Q-values).
    """

    x = self.layer1(x)
    x = torch.nn.functional.relu(x)
    x = self.layer2(x)
    x = torch.nn.functional.relu(x)
    output_tensor = self.layer3(x)
    
    return output_tensor

class MinimumExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
  def __init__(
    self,
    optimizer: torch.optim.Optimizer,
    lr_decay: float,
    last_epoch: int = -1,
    min_lr: float = 1e-6,
  ):
    """
    Initialize a new instance of MinimumExponentialLR.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose learning rate should be scheduled.
    lr_decay : float
        The multiplicative factor of learning rate decay.
    last_epoch : int, optional
        The index of the last epoch. Default is -1.
    min_lr : float, optional
        The minimum learning rate. Default is 1e-6.
    """
    self.min_lr = min_lr
    super().__init__(optimizer, lr_decay, last_epoch=-1)

  def get_lr(self) -> List[float]:
    """
    Compute learning rate using chainable form of the scheduler.

    Returns
    -------
    List[float]
        The learning rates of each parameter group.
    """
    return [
      max(base_lr * self.gamma**self.last_epoch, self.min_lr)
      for base_lr in self.base_lrs
    ]

class DeepQLearningAgent:
  def __init__(self,
               logger = None,
               device: str = "cuda",
               loss_fn: Callable = None,
               gamma: float = 0.9,
               min_epsilon: float = 0.05,
               max_epsilon: float = 0.3,
               decay_rate: float = 0.0001,
               learning_rate: float = 0.004):
    self.logger = logger
    self.device = device

    self.loss_fn = loss_fn or torch.nn.MSELoss()

    self.gamma = gamma
    self.min_epsilon = min_epsilon
    self.max_epsilon = max_epsilon
    self.decay_rate = decay_rate

    self.q_network = QNetwork(
      n_observations=20, # features vector has shape (20, )
      n_actions=len(ACTIONS),
      nn_l1=128,
      nn_l2=128
    )

    self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
    self.lr_scheduler = MinimumExponentialLR(self.optimizer, lr_decay=0.97, min_lr=0.0001)

  def act(self,
          feature_vector: list[int],
          n_round: int,
          train: bool = True) -> str:
    epsilon = self._compute_epsilon(n_round) if train else 0

    # epsilon-greedy strategy
    if random.random() < epsilon:
      action = random.choice(ACTIONS)
      if self.logger:
        self.logger.debug(f"Choosing {action} randomly")
      return action
    
    # otherwise, best action computed from our network
    action_idx = self._greedy_policy(feature_vector)

    if self.logger:
      self.logger.debug(f"Choosing {ACTIONS[action_idx]} from Q-network")
    return ACTIONS[action_idx]

  def _compute_epsilon(self, n_round: int) -> float:
    """Compute epsilon for the epsilon-greedy policy based on the round number."""
    return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * n_round)
  
  def _greedy_policy(self, feature_vector: tuple) -> int:
    """Select the action with the highest value from the Q-network. If more than one choose randomly."""
    # convert the state to a PyTorch tensor and add a batch dimension (unsqueeze)
    state_tensor = torch.tensor(feature_vector, dtype=torch.float32, device=self.device).unsqueeze(0)

    # compute the Q-values for the current state using the Q-network
    q_values = self.q_network(state_tensor)

    # find max value
    max_value = q_values.max()

    # select any action with max Q-value
    max_indices = (q_values.squeeze(0) == max_value).nonzero(as_tuple=True)[0]
    action_idx = max_indices[torch.randint(0, max_indices.size(0), (1,))].item()

    return action_idx

  def update_q_value(self,
                     state: list[int],
                     action_idx: int,
                     reward: float,
                     new_state: list[int]):
    done = 1 if new_state is None else 0

    with torch.no_grad():
      # Convert the next_state to a PyTorch tensor and add a batch dimension (unsqueeze)
      if not done:
        next_state_tensor = torch.tensor(new_state, dtype=torch.float32, device=self.device).unsqueeze(0)
      else:
        # all 0s, doesn't matter
        next_state_tensor = torch.zeros((1, len(state)), dtype=torch.float32, device=self.device)

      # Compute the **target** Q-value
      target  = torch.tensor(reward, device=self.device, dtype=torch.float32)
      target += torch.tensor(self.gamma * (1 - done), dtype=torch.float32) * self.q_network(next_state_tensor).squeeze(0).max()

    # Convert the state to a PyTorch tensor and add a batch dimension (unsqueeze)
    state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    # Compute the Q-values for the current state using the Q-network
    q_values = self.q_network(state_tensor)

    # Get the Q-value of the current action
    q_value_of_current_action = q_values.squeeze(0)[action_idx]
    
    # Loss & backprop.
    loss = self.loss_fn(q_value_of_current_action, target)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    self.lr_scheduler.step()

    if self.logger:
      self.logger.debug(f"Updated Action {action_idx}")

  def training_step(self,
                    state: list[int],
                    action: str,
                    reward: float,
                    new_state: list[int]):
    action_idx = ACTIONS.index(action)
    self.update_q_value(state, action_idx, reward, new_state)

  def save(self, model_name="dqn.pth"):
    model_path = os.path.join('./models', model_name)
    torch.save(self.q_network, model_path)
    if self.logger:
      self.logger.info("DQN saved to {}".format(model_path))