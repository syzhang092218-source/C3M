import torch
import torch.nn as nn

from .base import Controller

from network.mlp import MLP


class NeuralTrackingController(Controller):

    def __init__(
            self,
            state_dim,
            action_dim,
            goal_point: torch.Tensor,
            u_eq: torch.Tensor,
            state_std: torch.Tensor,
            ctrl_std: torch.Tensor,
            hidden_layers: tuple = (128, 128)
    ):
        super(NeuralTrackingController, self).__init__(goal_point, u_eq, state_std, ctrl_std)

        c = 3 * state_dim

        self.w1 = MLP(
            in_dim=state_dim*2,
            out_dim=c*state_dim,
            hidden_layers=hidden_layers,
            hidden_activation=nn.Tanh(),
        )

        self.w2 = MLP(
            in_dim=state_dim*2,
            out_dim=c*action_dim,
            hidden_layers=hidden_layers,
            hidden_activation=nn.Tanh(),
        )

        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, x: torch.Tensor, x_ref: torch.Tensor, u_ref: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.squeeze(-1)
        if x_ref.ndim == 3:
            x_ref = x_ref.squeeze(-1)

        x_trans = self.normalize_state(x)
        x_ref_trans = self.normalize_state(x_ref)
        # x_trans = x
        # x_ref_trans = x_ref

        bs = x_trans.shape[0]
        x_in = torch.cat((x_trans, x_ref_trans), dim=1)
        x_error = x_trans - x_ref_trans

        w1 = self.w1(x_in).view(bs, -1, self.state_dim)
        w2 = self.w2(x_in).view(bs, self.action_dim, -1)

        u = torch.bmm(w2, torch.tanh(torch.bmm(w1, x_error.unsqueeze(-1)))).squeeze(-1)

        return self.de_normalize_action(u).unsqueeze(-1) + u_ref
        # return u.unsqueeze(-1) + u_ref

    def act(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def track(self, x: torch.Tensor, x_ref: torch.Tensor, u_ref: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.squeeze(-1)
        if x_ref.ndim == 3:
            x_ref = x_ref.squeeze(-1)

        x_trans = self.normalize_state(x)
        x_ref_trans = self.normalize_state(x_ref)
        # x_trans = x
        # x_ref_trans = x_ref

        bs = x_trans.shape[0]
        x_in = torch.cat((x_trans, x_ref_trans), dim=1)
        x_error = x_trans - x_ref_trans

        with torch.no_grad():
            w1 = self.w1(x_in).view(bs, -1, self.state_dim)
            w2 = self.w2(x_in).view(bs, self.action_dim, -1)

            u = torch.bmm(w2, torch.tanh(torch.bmm(w1, x_error.unsqueeze(-1)))).squeeze(-1)

        return self.de_normalize_action(u) + u_ref
        # return u.unsqueeze(-1) + u_ref
