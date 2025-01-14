import torch
from torch import nn
from torch.autograd import grad
import numpy as np

from controller.neural_c3m_controller import NeuralC3MController


def get_model(num_dim_x, num_dim_control, w_lb, use_cuda = False):
    device = torch.device('cuda' if use_cuda else 'cpu')
    controller = NeuralC3MController(
        state_dim=4,
        action_dim=2,
        goal_point=torch.tensor([[0., 0., 0., 0.5]], device=device),
        u_eq=torch.tensor([0., 0.], device=device),
        state_std=1 / np.sqrt(12) * (torch.tensor([2., 2., 2 * np.pi, 1.], device=device) - torch.tensor([-2., -2., -2 * np.pi, 0.], device=device)),
        ctrl_std=1 / np.sqrt(12) * (torch.tensor([2., 0.3], device=device) - torch.tensor([-2., -0.3], device=device)),
    ).to(device)
    W_func = controller.W

    def u_func(x, xe, uref):
        return controller.u(x, x - xe, uref)

    return controller._W, 0, controller.controller.w1, controller.controller.w2, W_func, u_func, controller

effective_dim_start = 2 
effective_dim_end = 4

# class U_FUNC(nn.Module):
#     """docstring for U_FUNC."""
#
#     def __init__(self, model_u_w1, model_u_w2, num_dim_x, num_dim_control):
#         super(U_FUNC, self).__init__()
#         self.model_u_w1 = model_u_w1
#         self.model_u_w2 = model_u_w2
#         self.num_dim_x = num_dim_x
#         self.num_dim_control = num_dim_control
#
#     def forward(self, x, xe, uref):
#         # x: B x n x 1
#         # u: B x m x 1
#         bs = x.shape[0]
#
#         # w1 = self.model_u_w1(torch.cat([x[:,effective_dim_start:effective_dim_end,:],(x-xe)[:,effective_dim_start:effective_dim_end,:]],dim=1).squeeze(-1)).reshape(bs, -1, self.num_dim_x)
#         # w2 = self.model_u_w2(torch.cat([x[:,effective_dim_start:effective_dim_end,:],(x-xe)[:,effective_dim_start:effective_dim_end,:]],dim=1).squeeze(-1)).reshape(bs, self.num_dim_control, -1)
#         # u = w2.matmul(torch.tanh(w1.matmul(xe))) + uref
#
#         w1 = self.model_u_w1(torch.cat([x, (x - xe)], dim=1).squeeze(-1)).reshape(bs, -1, self.num_dim_x)
#         w2 = self.model_u_w2(torch.cat([x, (x - xe)], dim=1).squeeze(-1)).reshape(bs, self.num_dim_control, -1)
#         u = w2.matmul(torch.tanh(w1.matmul(xe))) + uref
#
#         return u
#
# def get_model(num_dim_x, num_dim_control, w_lb, use_cuda = False):
#     model_Wbot = torch.nn.Sequential(
#         torch.nn.Linear(1, 128, bias=True),
#         torch.nn.Tanh(),
#         torch.nn.Linear(128, (num_dim_x-num_dim_control) ** 2, bias=False))
#
#     # dim = effective_dim_end - effective_dim_start
#     dim = 4
#     model_W = torch.nn.Sequential(
#         torch.nn.Linear(dim, 128, bias=True),
#         torch.nn.Tanh(),
#         torch.nn.Linear(128, num_dim_x * num_dim_x, bias=False))
#
#     c = 3 * num_dim_x
#     model_u_w1 = torch.nn.Sequential(
#         torch.nn.Linear(2*dim, 128, bias=True),
#         torch.nn.Tanh(),
#         torch.nn.Linear(128, c*num_dim_x, bias=True))
#
#     model_u_w2 = torch.nn.Sequential(
#         torch.nn.Linear(2*dim, 128, bias=True),
#         torch.nn.Tanh(),
#         torch.nn.Linear(128, num_dim_control*c, bias=True))
#
#     if use_cuda:
#         model_W = model_W.cuda()
#         model_Wbot = model_Wbot.cuda()
#         model_u_w1 = model_u_w1.cuda()
#         model_u_w2 = model_u_w2.cuda()
#
#     def W_func(x):
#         bs = x.shape[0]
#         x = x.squeeze(-1)
#
#         # W = model_W(x[:,effective_dim_start:effective_dim_end]).view(bs, num_dim_x, num_dim_x)
#         # Wbot = model_Wbot(torch.ones(bs, 1).type(x.type())).view(bs, num_dim_x-num_dim_control, num_dim_x-num_dim_control)
#         # W[:, 0:num_dim_x-num_dim_control, 0:num_dim_x-num_dim_control] = Wbot
#         # W[:, num_dim_x-num_dim_control::, 0:num_dim_x-num_dim_control] = 0
#
#         W = model_W(x).view(bs, num_dim_x, num_dim_x)
#
#         W = W.transpose(1,2).matmul(W)
#         W = W + w_lb * torch.eye(num_dim_x).view(1, num_dim_x, num_dim_x).type(x.type())
#         return W
#
#
#     u_func = U_FUNC(model_u_w1, model_u_w2, num_dim_x, num_dim_control)
#
#     return model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func
