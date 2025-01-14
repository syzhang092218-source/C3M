import torch
import torch.nn as nn

from .base import Controller
from .neural_tracking_controller import NeuralTrackingController

from network.mlp import MLP


class NeuralC3MController(Controller):

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            goal_point: torch.Tensor,
            u_eq: torch.Tensor,
            state_std: torch.Tensor,
            ctrl_std: torch.Tensor,
            hidden_layers_c3m: tuple = (128, 128),
            hidden_layers_controller: tuple = (128, 128),
            w_lb: float = 0.1,
            w_ub: float = 10.0,
            c3m_lambda: float = 0.5
    ):
        super(NeuralC3MController, self).__init__(goal_point, u_eq, state_std, ctrl_std)

        # set up controller
        self.controller = NeuralTrackingController(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_point=goal_point,
            u_eq=u_eq,
            state_std=state_std,
            ctrl_std=ctrl_std,
            hidden_layers=hidden_layers_controller
        )

        # set up C3M
        self._W = MLP(
            in_dim=state_dim,
            out_dim=state_dim*state_dim,
            hidden_layers=hidden_layers_c3m,
            hidden_activation=nn.Tanh()
        )

        # record params
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.c3m_lambda = c3m_lambda
        self.w_lb = w_lb
        self.w_ub = w_ub

    def forward(self, x: torch.Tensor, x_ref: torch.Tensor, u_ref: torch.Tensor) -> torch.Tensor:
        return self.controller(x, x_ref, u_ref)

    def u(self, x: torch.Tensor, x_ref: torch.Tensor, u_ref: torch.Tensor) -> torch.Tensor:
        return self.controller(x, x_ref, u_ref)

    def act(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def W(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        if x.ndim == 3:
            x = x.squeeze(-1)
        # x_trans = self.normalize_state(x)
        x_trans = x

        W = self._W(x_trans).view(bs, self.state_dim, self.state_dim)
        W_sym = torch.bmm(W.transpose(1, 2), W)
        return W_sym + self.w_lb * torch.eye(self.state_dim).view(1, self.state_dim, self.state_dim).type_as(x)

    def g_bot(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        g_bot = torch.cat((torch.eye(self.state_dim - self.action_dim, self.state_dim - self.action_dim),
                           torch.zeros(self.action_dim, self.state_dim - self.action_dim))).type_as(x)
        return g_bot.repeat(bs, 1, 1)
