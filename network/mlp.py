import torch
import torch.nn as nn

from .utils import init_param


class MLP(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, hidden_layers: tuple,
                 hidden_activation: nn.Module = nn.ReLU(), output_activation: nn.Module = None,
                 init: bool = True, gain: float = 1.):
        super().__init__()

        # build MLP
        layers = []
        units = in_dim
        for next_units in hidden_layers:
            if init:
                layers.append(init_param(nn.Linear(units, next_units), gain=gain))
            else:
                layers.append(nn.Linear(units, next_units))
            layers.append(hidden_activation)
            units = next_units
        if init:
            layers.append(init_param(nn.Linear(units, out_dim), gain=gain))
        else:
            layers.append(nn.Linear(units, out_dim))
        if output_activation is not None:
            layers.append(output_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
