import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return x

class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.state_value_fc1 = nn.Linear(state_size, 32)
        self.state_value_fc2 = nn.Linear(32, 1)
        
        self.action_advantage_fc1 = nn.Linear(state_size, 32)
        self.action_advantage_fc2 = nn.Linear(32, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        state_value = self.state_value_fc1(state)
        state_value = F.relu(state_value)
        state_value = self.state_value_fc2(state_value)
        
        action_advantage = self.action_advantage_fc1(state)
        action_advantage = F.relu(action_advantage)
        action_advantage = self.action_advantage_fc2(action_advantage)
        
        return state_value + (action_advantage - action_advantage.mean())