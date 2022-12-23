import torch
from torch.autograd import Variable


class DQN:
    """Deep Q Neural Network class."""

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 64, lr: float = 0.0005
    ):
        self.criterion = torch.nn.MSELoss()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim * 2, action_dim),
        )
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.action_dim = action_dim

    def update(self, action, state, y):
        """Update the weights of the network given a training sample."""
        action_tensor = torch.tensor(action).view(-1, 1)
        q_s_pred = self.model(torch.tensor(state)).view(-1, self.action_dim)
        q_s_a_pred = q_s_pred.gather(1, action_tensor)
        loss = self.criterion(q_s_a_pred, torch.tensor(y))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        """Compute Q values for all actions using the DQL."""
        with torch.no_grad():
            return self.model(torch.tensor(state))
