import torch
import numpy as np
import os
import math
import matplotlib.pyplot as plt

from typing import Optional, Tuple
from scipy.io import loadmat
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy import rad2deg, deg2rad
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from .base import ControlAffineSystem
from .utils import scale3d, rotate3d

import model.env.aerobench as aerobench_loader  # type: ignore
from aerobench.highlevel.controlled_f16 import controlled_f16  # type: ignore
from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot  # type: ignore
from aerobench.lowlevel.low_level_controller import LowLevelController  # type: ignore
from aerobench.visualize.anim3d import get_script_path  # type: ignore
from aerobench.visualize import plot  # type: ignore


class F16Tracking(ControlAffineSystem):
    """
    F16 Tracking environment.

    The system has state
        x[0] = air speed, VT    (ft/sec)
        x[1] = angle of attack, alpha  (rad)
        x[2] = angle of sideslip, beta (rad)
        x[3] = roll angle, phi  (rad)
        x[4] = pitch angle, theta  (rad)
        x[5] = yaw angle, psi  (rad)
        x[6] = roll rate, P  (rad/sec)
        x[7] = pitch rate, Q  (rad/sec)
        x[8] = yaw rate, R  (rad/sec)
        x[9] = northward horizontal displacement, pn  (feet)
        x[10] = eastward horizontal displacement, pe  (feet)
        x[11] = altitude, h  (feet)
        x[12] = engine thrust dynamics lag state, pow
        x[13, 14, 15] = internal integrator states

    and control inputs, which are setpoints for a lower-level integrator
        u[0] = Z acceleration
        u[1] = stability roll rate
        u[2] = side acceleration + yaw rate (usually regulated to 0)
        u[3] = throttle command (0.0, 1.0)

    The system is parameterized by
        lag_error: the additive error in the engine lag state dynamics
    """

    # number of states and controls
    N_DIMS = 16
    N_CONTROLS = 4

    # state indices
    VT = 0  # airspeed
    ALPHA = 1  # angle of attack
    BETA = 2  # sideslip angle
    PHI = 3  # roll angle
    THETA = 4  # pitch angle
    PSI = 5  # yaw angle
    Proll = 6  # roll rate
    Q = 7  # pitch rate
    R = 8  # yaw rate
    POSN = 9  # northward displacement
    POSE = 10  # eastward displacement
    H = 11  # altitude
    POW = 12  # engine thrust dynamics lag state

    # control indices
    U_NZ = 0  # desired z acceleration
    U_SR = 1  # desired stability roll rate
    U_NYR = 2  # desired side acceleration + yaw rate
    U_THROTTLE = 3  # throttle command

    # max episode steps
    MAX_EPISODE_STEPS = 500

    # stable level
    STABLE_LEVEL = 1000

    # name of the states
    STATE_NAME = [
        'airspeed',
        'angle of attack',
        'sideslip angle',
        'roll angle',
        'pitch angle',
        'yaw angle',
        'roll rate',
        'pitch rate',
        'yaw rate',
        'northward displacement',
        'eastward displacement',
        'altitude',
        'engine thrust dynamics lag',
        'integral 1',
        'integral 2',
        'integral 3'
    ]

    def __init__(
            self,
            device: torch.device,
            dt: float = 0.02,
            params: Optional[dict] = None,
            controller_dt: Optional[float] = None,
    ):
        super(F16Tracking, self).__init__(device, dt, params, controller_dt)

        # Since we aren't using a linearized controller, we need to provide
        # some guess for a Lyapunov matrix
        self.P = torch.eye(self.n_dims)
        self._setpoint = torch.tensor([5000., 7500., 1600.], dtype=torch.float, device=self.device)

    def reset(self) -> torch.Tensor:
        initial_conditions = torch.tensor([
            (520.0, 560.0),  # vt
            (deg2rad(2.1215), deg2rad(2.1215)),  # alpha
            (-0.0, 0.0),  # beta
            (0.0, 0.0),  # phi
            (0.0, 0.0),  # theta
            (0.0, 0.0),  # psi
            (-0.5, 0.5),  # P
            (-0.5, 0.5),  # Q
            (-0.5, 0.5),  # R
            (-0.0, 0.0),  # PN
            (-0.0, 0.0),  # PE
            (1500.0, 1800.0),  # H
            (4.0, 5.0),  # pow
            (0.0, 0.0),  # integrator state 1
            (0.0, 0.0),  # integrator state 2
            (0.0, 0.0),  # integrator state 3
        ], dtype=torch.float, device=self.device)
        self._t = 0
        self._state = torch.rand(1, self.n_dims, device=self.device)
        self._state = self._state * (initial_conditions[:, 1] - initial_conditions[:, 0]) + initial_conditions[:, 0]
        return self.state

    def step(self, u: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        raise NotImplementedError

    def render(self) -> np.ndarray:
        raise NotImplementedError

    def render_demo(self, state: torch.Tensor, t: int, action: torch.Tensor = None) -> np.ndarray:
        raise NotImplementedError

    def default_param(self) -> dict:
        return {'lag_error': 0.0}

    def validate_params(self, params: dict) -> bool:
        valid = 'lag_error' in params

        return valid

    def state_name(self, dim: int) -> str:
        return F16Tracking.STATE_NAME[dim]

    @property
    def n_dims(self) -> int:
        return F16Tracking.N_DIMS

    @property
    def n_controls(self) -> int:
        return F16Tracking.N_CONTROLS

    @property
    def max_episode_steps(self) -> int:
        return F16Tracking.MAX_EPISODE_STEPS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        upper_limit = torch.tensor([6.0, 20.0, 20.0, 1.0], device=self.device)
        lower_limit = torch.tensor([-1.0, -20.0, -20.0, 0.0], device=self.device)

        return upper_limit, lower_limit

    def _f(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _g(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def control_affine_dynamics(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.params

        # The f16 model is not batched, so we need to compute f and g for each row of x
        n_batch = x.shape[0]
        f = torch.zeros((n_batch, self.n_dims, 1)).type_as(x)
        g = torch.zeros(n_batch, self.n_dims, self.n_controls).type_as(x)

        # Convert input to numpy
        x = x.detach().cpu().numpy()
        for batch in range(n_batch):
            # Get the derivatives at each of n_controls + 1 linearly independent points
            # (plus zero) to fit control-affine dynamics
            u = np.zeros((1, self.n_controls))
            for i in range(self.n_controls):
                u_i = np.zeros((1, self.n_controls))
                u_i[0, i] = 1.0
                u = np.vstack((u, u_i))

            # Compute derivatives at each of these points
            llc = LowLevelController()
            model = "stevens"  # look-up table
            # model = "morelli"  # polynomial fit
            t = 0.0
            xdot = np.zeros((self.n_controls + 1, self.n_dims))
            for i in range(self.n_controls + 1):
                xdot[i, :], _, _, _, _ = controlled_f16(
                    t, x[batch, :], u[i, :], llc, f16_model=model
                )

            # Run a least-squares regression to fit control-affine dynamics
            # We want a relationship of the form
            #       xdot = f(x) + g(x)*u, or xdot = [f, g]*[1, u]
            # Augment the inputs with a one column for the control-independent part
            regressors = np.hstack((np.ones((self.n_controls + 1, 1)), u))
            # Compute the least-squares fit and find A^T such that xdot = [1, u] A^T
            A, residuals, _, _ = np.linalg.lstsq(regressors, xdot, rcond=None)
            A = A.T
            # Extract the control-affine fit
            f[batch, :, 0] = torch.tensor(A[:, 0]).type_as(f)
            g[batch, :, :] = torch.tensor(A[:, 1:]).type_as(g)

            # Add in the lag error (which we're treating as bounded additive error)
            f[batch, self.POW] += params["lag_error"]

        return f, g

    def closed_loop_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # The F16 model is not batched, so we need to derivatives for each x separately
        n_batch = x.size()[0]
        xdot = torch.zeros_like(x).type_as(x)

        # Convert input to numpy
        x_np = x.detach().cpu().numpy()
        u_np = u.detach().cpu().numpy()
        for batch in range(n_batch):
            # Compute derivatives at this point
            llc = LowLevelController()
            model = "stevens"  # look-up table
            # model = "morelli"  # polynomial fit
            t = 0.0
            xdot_np, _, _, _, _ = controlled_f16(
                t, x_np[batch, :], u_np[batch, :], llc, f16_model=model
            )

            xdot[batch, :] = torch.tensor(xdot_np).type_as(x)

        return xdot

    def u_nominal(self, x: torch.Tensor) -> torch.Tensor:
        setpoint = self._setpoint.cpu().numpy()
        pilot = WaypointAutopilot([tuple(setpoint)], stdout=True)

        raise NotImplementedError

    @property
    def use_lqr(self):
        return False

    @property
    def goal_point(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def state(self) -> torch.Tensor:
        if self._state is not None:
            while self._state[0, 5] > np.pi:
                self._state[0, 5] -= np.pi
            while self._state[0, 5] < -np.pi:
                self._state[0, 5] += np.pi
            while self._state[0, 3] > np.pi:
                self._state[0, 3] -= np.pi
            while self._state[0, 3] < -np.pi:
                self._state[0, 3] += np.pi
            return self._state.squeeze(0)
        else:
            raise ValueError('State is not initialized')
