### small modification of NeusScheduler

from dataclasses import dataclass, field
from typing import Any, Optional, Type, List

import numpy as np
from torch.optim import Optimizer, lr_scheduler

from nerfstudio.configs.base_config import InstantiateConfig



@dataclass
class NeuSSchedulerConfig(InstantiateConfig):
    """Basic scheduler config with self-defined exponential decay schedule"""

    _target: Type = field(default_factory=lambda: NeuSScheduler)
    warm_up_start: int = 0
    warm_up_end: int = 5000
    learning_rate_alpha: float = 0.05
    max_steps: int = 300000
    end: int = 300000

    def setup(self, optimizer=None, lr_init=None, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(
            optimizer,
            self.warm_up_start,
            self.warm_up_end,
            self.learning_rate_alpha,
            self.max_steps,
            self.end
        )


class NeuSScheduler(lr_scheduler.LambdaLR):
    """Starts with a flat lr schedule until it reaches N epochs then applies a given scheduler"""

    def __init__(self, optimizer, warm_up_start, warm_up_end, learning_rate_alpha, max_steps, end) -> None:
        def func(step):
            if step < warm_up_start:
                learning_factor = 0.
            elif step < warm_up_end:
                learning_factor = (step - warm_up_start) / (warm_up_end- warm_up_start)
            elif step < end:
                alpha = learning_rate_alpha
                learning_factor = alpha ** ((step - warm_up_end) / max_steps)
            else:
                learning_factor = 0.
            return learning_factor

        super().__init__(optimizer, lr_lambda=func)