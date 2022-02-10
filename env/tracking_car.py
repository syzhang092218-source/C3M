import copy
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Tuple
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from .base import ControlAffineSystem
from .utils import get_rectangle, plot_poly, lqr


class VehicleParameters(object):
    """A restricted set of the commonroad vehicle parameters, which can be
    found as forked at https://github.com/dawsonc/commonroad-vehicle-models"""

    def __init__(self):
        super(VehicleParameters, self).__init__()

        self.steering_min = -1.066  # minimum steering angle [rad]
        self.steering_max = 1.066  # maximum steering angle [rad]
        self.steering_v_min = -0.4  # minimum steering velocity [rad/s]
        self.steering_v_max = 0.4  # maximum steering velocity [rad/s]

        self.longitudinal_a_max = 11.5  # maximum absolute acceleration [m/s^2]

        self.tire_p_dy1 = 1.0489  # Lateral friction Muy
        self.tire_p_ky1 = -21.92  # Maximum value of stiffness Kfy/Fznom

        # distance from spring mass center of gravity to front axle [m]  LENA
        self.a = 0.3048 * 3.793293
        # distance from spring mass center of gravity to rear axle [m]  LENB
        self.b = 0.3048 * 4.667707
        self.h_s = 0.3048 * 2.01355  # M_s center of gravity above ground [m]  HS
        self.m = 4.4482216152605 / 0.3048 * 74.91452  # vehicle mass [kg]  MASS
        # moment of inertia for sprung mass in yaw [kg m^2]  IZZ
        self.I_z = 4.4482216152605 * 0.3048 * 1321.416


class TrackingCar(ControlAffineSystem):
    """
    Represents a car using the kinematic single-track model.

    The system has state defined relative to a reference path
    [x_ref, y_ref, psi_ref, v_ref, omega_ref, a_ref]

        x = [s_x - x_ref, s_y - y_ref, delta, v - v_ref, psi - psi_ref, omega_ref, t]

    where s_x and s_y are the x and y position, delta is the steering angle, v is the
    longitudinal velocity, and psi is the heading. The errors in x and y are expressed
    in the reference path frame

    The control inputs are

        u = [v_delta, a_long]

    representing the steering effort (change in delta) and longitudinal acceleration.

    The system is parameterized by a bunch of car-specific parameters, which we load
    from the commonroad model, and by the parameters of the reference point. Instead of
    viewing these as time-varying parameters, we can view them as bounded uncertainties,
    particularly in omega_ref and a_ref.
    """

    # number of states and controls
    N_DIMS = 7
    N_CONTROLS = 2

    # state indices
    SXE = 0
    SYE = 1
    DELTA = 2
    VE = 3
    PSI_E = 4
    OMEGA_REF = 5
    T = 6

    # control indices
    VDELTA = 0
    ALONG = 1

    # max episode steps
    MAX_EPISODE_STEPS = 1000

    # name of the states
    STATE_NAME = [
        'x error',
        'y error',
        'steering angle'
        'velocity'
        'heading error'
    ]

    def __init__(
            self,
            device: torch.device,
            dt: float = 0.01,
            params: Optional[dict] = None,
            controller_dt: Optional[float] = None,
    ):
        # Get car parameters
        self.car_params = VehicleParameters()

        super(TrackingCar, self).__init__(device, dt, params, controller_dt)

        self._ref_path = self._generate_ref(self.params["path"])
        self._info = copy.deepcopy(self.params)

    def _set_info(self, t: int) -> dict:
        info = copy.deepcopy(self.params)
        while t >= TrackingCar.MAX_EPISODE_STEPS:
            t -= TrackingCar.MAX_EPISODE_STEPS
        info['x_ref'] = self._ref_path[t, 0]
        info['y_ref'] = self._ref_path[t, 1]
        info['v_ref'] = self._ref_path[t, 2]
        info['psi_ref'] = self._ref_path[t, 3]
        info['omega_ref'] = self._ref_path[t, 4]
        return info

    def _generate_ref(self, path: str) -> torch.Tensor:
        ref_path = torch.zeros(self.max_episode_steps, 5, device=self.device)

        if path == 'circle':
            psi_ref = 0.0
            x_ref = 0.0
            y_ref = 0.0
            omega_ref = 2 * np.pi / (self.max_episode_steps * self.dt)
            for step in range(self.max_episode_steps):
                psi_ref += self.dt * omega_ref
                x_ref += self.dt * self.params["v_ref"] * np.cos(psi_ref)
                y_ref += self.dt * self.params["v_ref"] * np.sin(psi_ref)
                ref_path[step, :].copy_(torch.tensor([x_ref, y_ref, self.params["v_ref"], psi_ref, omega_ref]))
        elif path == 'sin':
            psi_ref = 1.0
            x_ref = 0.0
            y_ref = 0.0
            for step in range(self.max_episode_steps):
                omega_ref = 1.5 * np.sin(step * self.dt)
                psi_ref += self.dt * omega_ref
                x_ref += self.dt * self.params["v_ref"] * np.cos(psi_ref)
                y_ref += self.dt * self.params["v_ref"] * np.sin(psi_ref)
                ref_path[step, :].copy_(torch.tensor([x_ref, y_ref, self.params["v_ref"], psi_ref, omega_ref]))
        else:
            raise NotImplementedError('Unknown path')

        return ref_path

    def plot_ref(self, path: str = None):
        height = 5
        width = 5
        fig = plt.figure(figsize=(height, width))
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ref_path = self._ref_path.cpu().detach().numpy()
        x = ref_path[:, 0]
        y = ref_path[:, 1]
        plt.plot(x, y)
        ax.axis('off')
        if path is not None:
            plt.savefig(os.path.join(path, 'ref_path.png'))
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape((height * 100, width * 100, 3))
        plt.close()
        return data

    def reset(self) -> torch.Tensor:
        self._t = 0
        self._info = self._set_info(t=0)
        sxe = (torch.rand(1, 1, device=self.device) - 0.5) * 0.2
        sye = (torch.rand(1, 1, device=self.device) - 0.5) * 0.2
        delta = (torch.rand(1, 1, device=self.device) - 0.5) * 0.2
        ve = (torch.rand(1, 1, device=self.device) - 0.5) * 0.2
        psi_e = (torch.rand(1, 1, device=self.device) - 0.5) * 0.2
        omega_ref = torch.tensor([self._info['omega_ref']]).type_as(sxe).unsqueeze(0)
        t = torch.tensor([self._t]).type_as(sxe).unsqueeze(0)
        self._state = torch.cat((sxe, sye, delta, ve, psi_e, omega_ref, t), dim=1)
        return self.state

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
        reward = float(2.0 - torch.norm(self._state[0, :5]))
        return self.state, reward, done, self._info

    def render(self) -> np.ndarray:
        # plot background
        h = 500
        w = 500
        fig, ax = plt.subplots(figsize=(h / 100, w / 100), dpi=60)
        canvas = FigureCanvas(fig)
        ref_path = self._ref_path.cpu().detach().numpy()
        x = ref_path[:, 0]
        y = ref_path[:, 1]
        plt.plot(x, y)
        x_max, x_min = np.max(x), np.min(x)
        y_max, y_min = np.max(y), np.min(y)

        # extract state
        state = self.state
        x_car = state[0] + self._info['x_ref']
        y_car = state[1] + self._info['y_ref']
        psi_car = state[4] + self._info['psi_ref']
        length = 2
        width = 1.5

        # plot vehicle
        car = get_rectangle(torch.tensor([x_car, y_car]), float(psi_car), length, width)
        plot_poly(ax, car, 'red', alpha=0.7)
        plt.xlim((x_min - 3, x_max + 3))
        plt.ylim((y_min - 3, y_max + 3))

        # text
        text_point = (x_max - (x_max - x_min) / 4 + 2.5, y_max + 2)
        line_gap = (y_max - y_min) / 20
        plt.text(text_point[0], text_point[1], f'T: {self._t}')
        if self._action is not None:
            plt.text(text_point[0], text_point[1] - line_gap, f'delta: {self._action[0, 1]:.2f}')
            plt.text(text_point[0], text_point[1] - 2 * line_gap, f'a: {self._action[0, 0]:.2f}')

        # get rgb array
        ax.axis('off')
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return data

    def render_demo(self, state: torch.Tensor, t: int, action: torch.Tensor = None) -> np.ndarray:
        h = 500
        w = 500
        fig = plt.figure(figsize=(h / 100, w / 100), dpi=60)
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        # plot reference
        ref_path = self._ref_path.cpu().detach().numpy()
        x = ref_path[:, 0]
        y = ref_path[:, 1]
        plt.plot(x, y)
        x_max, x_min = np.max(x), np.min(x)
        y_max, y_min = np.max(y), np.min(y)

        # plot vehicle
        self._info = self._set_info(t)
        x_car = state[0, 0] + self._info['x_ref']
        y_car = state[0, 1] + self._info['y_ref']
        psi_car = state[0, 4] + self._info['psi_ref']
        length = 2
        width = 1.5
        car = get_rectangle(torch.tensor([x_car, y_car]), float(psi_car), length, width)
        plot_poly(ax, car, 'red', alpha=0.7)
        plt.xlim((x_min - 3, x_max + 3))
        plt.ylim((y_min - 3, y_max + 3))

        # text
        text_point = (x_max - (x_max - x_min) / 4 + 2.5, y_max + 2)
        line_gap = (y_max - y_min) / 20
        plt.text(text_point[0], text_point[1], f'T: {t}')
        if action is not None:
            plt.text(text_point[0], text_point[1] - line_gap, f'delta: {action[0, 1]:.2f}')
            plt.text(text_point[0], text_point[1] - 2 * line_gap, f'a: {action[0, 0]:.2f}')

        # get rgb array
        ax.axis('off')
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return data

    def default_param(self) -> dict:
        return {
            "psi_ref": torch.tensor(1.0, device=self.device),
            "v_ref": torch.tensor(10.0, device=self.device),
            "a_ref": torch.tensor(0.0, device=self.device),
            "omega_ref": torch.tensor(0.0, device=self.device),
            "path": 'circle'
        }

    def validate_params(self, params: dict) -> bool:
        """
        Check if a given set of parameters is valid.

        Parameters
        ----------
        params: dict
            parameter values for the system.
            Requires keys ["psi_ref", "v_ref", "a_ref", "omega_ref"]

        Returns
        -------
        valid: bool
            True if parameters are valid, False otherwise
        """
        valid = True

        # make sure all needed parameters were provided
        valid = valid and "psi_ref" in params
        valid = valid and "v_ref" in params
        valid = valid and "a_ref" in params
        valid = valid and "omega_ref" in params
        valid = valid and "path" in params

        return valid

    def state_name(self, dim: int) -> str:
        return TrackingCar.STATE_NAME[dim]

    @property
    def n_dims(self) -> int:
        return TrackingCar.N_DIMS

    @property
    def n_controls(self) -> int:
        return TrackingCar.N_CONTROLS

    @property
    def max_episode_steps(self) -> int:
        return TrackingCar.MAX_EPISODE_STEPS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        upper_limit = torch.ones(self.n_dims, device=self.device)
        upper_limit[TrackingCar.SXE] = 3.0
        upper_limit[TrackingCar.SYE] = 3.0
        upper_limit[TrackingCar.DELTA] = self.car_params.steering_max
        upper_limit[TrackingCar.VE] = 3.0
        upper_limit[TrackingCar.PSI_E] = np.pi / 2
        upper_limit[TrackingCar.OMEGA_REF] = 2.0
        upper_limit[TrackingCar.T] = 1000

        lower_limit = -1.0 * upper_limit
        lower_limit[TrackingCar.DELTA] = self.car_params.steering_min
        lower_limit[TrackingCar.T] = 0

        return upper_limit, lower_limit

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        upper_limit = 10 * torch.tensor(
            [
                5.0,  # self.car_params.steering_v_max,
                self.car_params.longitudinal_a_max,
            ],
            device=self.device
        )
        lower_limit = 10 * torch.tensor(
            [
                -5.0,  # self.car_params.steering_v_min,
                -self.car_params.longitudinal_a_max,
            ],
            device=self.device
        )

        return upper_limit, lower_limit

    def _f(self, x: torch.Tensor) -> torch.Tensor:
        # extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)

        # extract the parameters
        # v_ref = torch.tensor(self.params["v_ref"])
        # a_ref = torch.tensor(self.params["a_ref"])
        # omega_ref = torch.tensor(self.params["omega_ref"])
        v_ref = self._info["v_ref"]
        a_ref = self._info["a_ref"]

        # extract the state variables and adjust for the reference
        v = x[:, TrackingCar.VE] + v_ref
        psi_e = x[:, TrackingCar.PSI_E]
        delta = x[:, TrackingCar.DELTA]
        sxe = x[:, TrackingCar.SXE]
        sye = x[:, TrackingCar.SYE]
        omega_ref = x[:, TrackingCar.OMEGA_REF]
        t = x[:, TrackingCar.T].type(torch.int)

        # get info of next time step
        omega_ref_next = torch.zeros(batch_size).type_as(omega_ref)
        for i in range(batch_size):
            omega_ref_next[i] = self._set_info(t[i] + 1)['omega_ref']

        # compute the dynamics
        wheelbase = self.car_params.a + self.car_params.b

        # We want to express the error in x and y in the reference path frame, so
        # we need to get the dynamics of the rotated global frame error
        dsxe_r = v * torch.cos(psi_e) - v_ref + omega_ref * sye
        dsye_r = v * torch.sin(psi_e) - omega_ref * sxe

        f[:, TrackingCar.SXE, 0] = dsxe_r
        f[:, TrackingCar.SYE, 0] = dsye_r
        f[:, TrackingCar.VE, 0] = -a_ref
        f[:, TrackingCar.DELTA, 0] = 0.0
        f[:, TrackingCar.PSI_E, 0] = v / wheelbase * torch.tan(delta) - omega_ref
        f[:, TrackingCar.OMEGA_REF, 0] = (omega_ref_next - omega_ref) / self.dt
        f[:, TrackingCar.T, 0] = 1. / self.dt

        return f

    def _g(self, x: torch.Tensor) -> torch.Tensor:
        # extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls))
        g = g.type_as(x)

        g[:, TrackingCar.DELTA, TrackingCar.VDELTA] = 1.0
        g[:, TrackingCar.VE, TrackingCar.ALONG] = 1.0

        return g

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(x.shape[0]).type_as(x)
        dist = torch.norm((x[:, :5] - self.goal_point[:, :5]), dim=1)
        for i in range(dist.shape[0]):
            if dist[i] > 0.01:
                mask[i] = 0
        return mask.bool()

    def u_nominal(self, x: torch.Tensor) -> torch.Tensor:
        params = self._info
        if x.ndim == 1:
            y = x[:5]
        else:
            y = x[:, :5]

        # Compute the LQR gain matrix

        # Linearize the system about the path
        wheelbase = self.car_params.a + self.car_params.b
        x0 = self.goal_point[0, :5].unsqueeze(0)
        x0[0, TrackingCar.DELTA] = torch.atan(
            params["omega_ref"] * wheelbase / params["v_ref"]
        )
        x0 = x0.type_as(x)
        A = np.zeros((self.n_dims - 2, self.n_dims - 2))
        A[TrackingCar.SXE, TrackingCar.SYE] = self.params["omega_ref"]
        A[TrackingCar.SXE, TrackingCar.VE] = 1

        A[TrackingCar.SYE, TrackingCar.SXE] = -self.params["omega_ref"]
        A[TrackingCar.SYE, TrackingCar.PSI_E] = self.params["v_ref"]

        A[TrackingCar.PSI_E, TrackingCar.VE] = torch.tan(x0[0, TrackingCar.DELTA]) / wheelbase
        A[TrackingCar.PSI_E, TrackingCar.DELTA] = self.params["v_ref"] / wheelbase

        A = np.eye(self.n_dims - 2) + self.controller_dt * A

        B = np.zeros((self.n_dims - 2, self.n_controls))
        B[TrackingCar.DELTA, TrackingCar.VDELTA] = 1.0
        B[TrackingCar.VE, TrackingCar.ALONG] = 1.0
        B = self.controller_dt * B

        # Define cost matrices as identity
        Q = np.eye(self.n_dims - 2)
        R = np.eye(self.n_controls)

        # Get feedback matrix
        self._K = torch.tensor(lqr(A, B, Q, R))

        # Compute nominal control from feedback + equilibrium control
        u_nominal = -(self._K.type_as(x) @ (y - x0).T).T
        u_eq = torch.zeros_like(u_nominal)
        u = u_nominal + u_eq

        # Clamp given the control limits
        upper_u_lim, lower_u_lim = self.control_limits
        for dim_idx in range(self.n_controls):
            u[:, dim_idx] = torch.clamp(
                u[:, dim_idx],
                min=lower_u_lim[dim_idx].item(),
                max=upper_u_lim[dim_idx].item(),
            )

        return u

    @property
    def use_lqr(self):
        return False

    @property
    def goal_point(self) -> torch.Tensor:
        goal_point = torch.zeros((1, self.n_dims), device=self.device)
        goal_point[0, TrackingCar.T] = TrackingCar.MAX_EPISODE_STEPS / 2
        goal_point[0, TrackingCar.OMEGA_REF] = torch.mean(self._ref_path[:, 4])
        return goal_point

    def sample_states(self, batch_size: int) -> torch.Tensor:
        high, low = self.state_limits
        rand = torch.rand(batch_size, self.n_dims, device=self.device)
        states = rand * (high - low) + low

        states[:, TrackingCar.T] = states[:, TrackingCar.T].type(torch.long)
        for i in range(batch_size):
            states[i, TrackingCar.OMEGA_REF] = self._set_info(int(states[i, TrackingCar.T]))['omega_ref']

        return states
