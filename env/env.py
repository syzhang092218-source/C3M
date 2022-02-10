import torch

from typing import Optional

from .base import ControlAffineSystem
from .inverted_pendulum import InvertedPendulum
from .tracking_car import TrackingCar
from .neural_lander import NeuralLander
from .f16_gcas import F16GCAS
from .dubins_car import DubinsCarTracking


def make_env(
        env_id: str,
        device: torch.device = torch.device('cpu'),
) -> ControlAffineSystem:
    if env_id == 'InvertedPendulum':
        return InvertedPendulum(device)
    elif env_id == 'TrackingCarCircle':
        return TrackingCar(device)
    elif env_id == 'TrackingCarSin':
        env = TrackingCar(device)
        params = env.default_param()
        params['path'] = 'sin'
        return TrackingCar(device, params=params)
    elif env_id == 'NeuralLander':
        return NeuralLander(device)
    elif env_id == 'F16GCAS':
        return F16GCAS(device)
    elif env_id == 'DubinsCarTracking':
        return DubinsCarTracking(device)
    else:
        raise NotImplementedError(f'{env_id} not implemented')
