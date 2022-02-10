import numpy as np
import torch
import copy
import matplotlib.pyplot as plt

from typing import Tuple, Optional
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from .base import ControlAffineSystem
from .utils import get_rectangle, plot_poly


class DubinsCarTracking(ControlAffineSystem):
    # number of states and controls
    N_DIMS = 4
    N_CONTROLS = 2

    # state indices
    X = 0
    Y = 1
    THETA = 2
    V = 3

    # control indices
    OMEGA = 0
    ALONG = 1

    # max episode steps
    MAX_EPISODE_STEPS = 1000

    # name of the states
    STATE_NAME = [
        'x',
        'y',
        'heading'
        'velocity'
    ]

    def __init__(
            self,
            device: torch.device,
            dt: float = 0.01,
            params: Optional[dict] = None,
            controller_dt: Optional[float] = None
    ):
        super(DubinsCarTracking, self).__init__(device, dt, params, controller_dt)

        self._ref_path = self._generate_ref(self.params["path"])
        self._info = copy.deepcopy(self.params)

    def _set_info(self, t: int) -> dict:
        info = {}
        while t >= DubinsCarTracking.MAX_EPISODE_STEPS:
            t -= DubinsCarTracking.MAX_EPISODE_STEPS
        info['x_ref'] = self._ref_path[t, 0]
        info['y_ref'] = self._ref_path[t, 1]
        info['theta_ref'] = self._ref_path[t, 2]
        info['v_ref'] = self._ref_path[t, 3]
        info['omega_ref'] = self._ref_path[t, 4]
        info['a_ref'] = self._ref_path[t, 5]
        return info

    def _generate_ref(self, path: str) -> torch.Tensor:
        ref_path = torch.zeros(self.max_episode_steps, 6, device=self.device)

        if path == 's':
            theta_ref = 0.0
            x_ref = 0.0
            y_ref = 0.0
            v_ref = 0.1
            for step in range(self.max_episode_steps):
                if step < self.max_episode_steps / 2:
                    a_ref = 0.1
                else:
                    a_ref = -0.1
                omega_ref = 1.5 * np.sin(step * self.dt)
                theta_ref += self.dt * omega_ref
                v_ref += a_ref * self.dt
                x_ref += self.dt * v_ref * np.cos(theta_ref)
                y_ref += self.dt * v_ref * np.sin(theta_ref)
                ref_path[step, :].copy_(torch.tensor([x_ref, y_ref, theta_ref, v_ref, omega_ref, a_ref]))
        else:
            raise NotImplementedError('Dubins car error: Unknown path')

        return ref_path

    def reset(self) -> torch.Tensor:
        self._t = 0
        self._info = self._set_info(t=0)
        initial_x = torch.tensor([
            (-0.2, 0.2),
            (-0.2, 0.2),
            (-0.5, 0.5),
            (0.0, 0.2),
        ])
        self._state = torch.rand(1, self.n_dims, device=self.device)
        self._state = self._state * (initial_x[:, 1] - initial_x[:, 0]) + initial_x[:, 0]
        return self.state

    def get_ref(self):
        ref_path = self._ref_path[self._t, :]
        return ref_path[:4], ref_path[4:]

    def step(self, u: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        if u.ndim == 1:
            u = u.unsqueeze(0)

        # clamp given the control limits
        upper_u_lim, lower_u_lim = self.control_limits
        for dim_idx in range(self.n_controls):
            u[:, dim_idx] = torch.clamp(
                u[:, dim_idx],
                min=lower_u_lim[dim_idx].item(),
                max=upper_u_lim[dim_idx].item(),
            )

        # reset reference to the next time
        self._t += 1
        self._info = self._set_info(t=self._t)

        # calculate returns
        self._state = self.forward(self._state, u)
        self._action = u
        upper_x_lim, lower_x_limit = self.state_limits
        done = self._t >= self.max_episode_steps or \
               (self._state > upper_x_lim).any() or (self._state < lower_x_limit).any()
        xy_ref = torch.cat((self._info['x_ref'].unsqueeze(0), self._info['y_ref'].unsqueeze(0)))
        error = self.state[:2] - xy_ref
        reward = float(2.0 - torch.norm(error))
        return self.state, reward, done, self._info

    def render(self) -> np.ndarray:
        # plot background
        h = 1000
        w = 1000
        fig, ax = plt.subplots(figsize=(h / 100, w / 100), dpi=100)
        canvas = FigureCanvas(fig)
        ref_path = self._ref_path.cpu().detach().numpy()
        x = ref_path[:, 0]
        y = ref_path[:, 1]
        plt.plot(x, y)
        state_limits = self.state_limits
        x_max, x_min = state_limits[0][0], state_limits[1][0]
        y_max, y_min = state_limits[0][1], state_limits[1][1]

        # extract state
        state = self.state
        x_car = state[0]
        y_car = state[1]
        theta_car = state[2]
        length = 0.1
        width = 0.07

        # plot vehicle
        car = get_rectangle(torch.tensor([x_car, y_car]), float(theta_car), length, width)
        plot_poly(ax, car, 'red', alpha=0.7)

        # plot reference vehicle
        info = self._info
        car_ref = get_rectangle(
            torch.tensor([info['x_ref'], info['y_ref']]), float(info['theta_ref']), length / 2, width / 2)
        plot_poly(ax, car_ref, 'blue', alpha=0.5)

        plt.xlim((x_min, x_max))
        plt.ylim((y_min, y_max))

        # text
        text_point = (x_max - (x_max - x_min) / 4 + 2.5, y_max + 2)
        line_gap = (y_max - y_min) / 20
        plt.text(text_point[0], text_point[1], f'T: {self._t}')
        if self._action is not None:
            plt.text(text_point[0], text_point[1] - line_gap, f'omega: {self._action[0, 1]:.2f}')
            plt.text(text_point[0], text_point[1] - 2 * line_gap, f'along: {self._action[0, 0]:.2f}')

        # get rgb array
        ax.axis('off')
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return data

    def render_demo(self, state: torch.Tensor, t: int, action: torch.Tensor = None) -> np.ndarray:
        raise NotImplementedError

    def default_param(self) -> dict:
        return {
            'path': 's'
        }

    def validate_params(self, params: dict) -> bool:
        valid = True

        # make sure all needed parameters were provided
        valid = valid and 'path' in params

        return valid

    def state_name(self, dim: int) -> str:
        return DubinsCarTracking.STATE_NAME[dim]

    def _f(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        f = torch.zeros((bs, self.n_dims, 1)).type_as(x)

        # extract states
        v = x[:, DubinsCarTracking.V]
        theta = x[:, DubinsCarTracking.THETA]

        f[:, DubinsCarTracking.X, 0] = v * torch.cos(theta)
        f[:, DubinsCarTracking.Y, 0] = v * torch.sin(theta)

        return f

    def _g(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        g = torch.zeros((bs, self.n_dims, self.n_controls)).type_as(x)

        g[:, DubinsCarTracking.THETA, DubinsCarTracking.OMEGA] = 1.0
        g[:, DubinsCarTracking.V, DubinsCarTracking.ALONG] = 1.0

        return g

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def n_dims(self) -> int:
        return DubinsCarTracking.N_DIMS

    @property
    def n_controls(self) -> int:
        return DubinsCarTracking.N_CONTROLS

    @property
    def max_episode_steps(self) -> int:
        return DubinsCarTracking.MAX_EPISODE_STEPS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        upper_limit = torch.ones(self.n_dims, device=self.device)
        upper_limit[DubinsCarTracking.X] = 2.0
        upper_limit[DubinsCarTracking.Y] = 2.0
        upper_limit[DubinsCarTracking.THETA] = 2 * np.pi
        upper_limit[DubinsCarTracking.V] = 1.0

        lower_limit = -1.0 * upper_limit
        lower_limit[DubinsCarTracking.V] = 0.0

        return upper_limit, lower_limit

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        upper_limit = torch.ones(self.n_controls, device=self.device)
        upper_limit[DubinsCarTracking.OMEGA] = 2.0
        upper_limit[DubinsCarTracking.ALONG] = 0.3

        lower_limit = -1.0 * upper_limit

        return upper_limit, lower_limit

    @property
    def goal_point(self) -> torch.Tensor:
        goal_point = torch.zeros((1, self.n_dims), device=self.device)
        goal_point[0, DubinsCarTracking.V] = 0.5
        return goal_point
    #
    # def u_nominal(self, x: torch.Tensor) -> torch.Tensor:
    #
    #
    # @property
    # def use_lqr(self):
    #     return False
