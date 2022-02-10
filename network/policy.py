import torch
import torch.nn as nn

from typing import Tuple

from .mlp import MLP
from .utils import reparameterize, evaluate_log_pi


class StateIndependentPolicy(nn.Module):
    """
    Stochastic policy \pi(a|s)
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_units: tuple = (128, 128),
            hidden_activation: nn.Module = nn.Tanh()
    ):
        super().__init__()

        self.net = MLP(
            in_dim=state_dim,
            out_dim=action_dim,
            hidden_layers=hidden_units,
            hidden_activation=hidden_activation,
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Get the mean of the stochastic policy

        Parameters
        ----------
        states: torch.Tensor
            input states

        Returns
        -------
        actions: torch.Tensor
            mean of the stochastic policy
        """
        return self.net(states)

    def sample(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions given states

        Parameters
        ----------
        states: torch.Tensor
            input states

        Returns
        -------
        actions: torch.Tensor
            actions to take
        log_pi: torch.Tensor
            log_pi of the actions
        """
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the log(\pi(a|s)) of the given action

        Parameters
        ----------
        states: torch.Tensor
            states that the actions act in
        actions: torch.Tensor
            actions taken

        Returns
        -------
        log_pi: : torch.Tensor
            log(\pi(a|s))
        """
        return evaluate_log_pi(self.net(states), self.log_stds, actions)
