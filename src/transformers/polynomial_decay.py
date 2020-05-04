from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class PolynomialDecayScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        learning_rate: float,
        end_learning_rate: float,
        num_training_steps: int,
        last_epoch: int = -1,
        power: float = 1.0,
    ) -> None:
        self._optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.learning_rate = learning_rate
        self.end_learning_rate = end_learning_rate
        self.num_training_steps = num_training_steps
        self.power = power
        self.last_step = last_epoch

        if self.num_warmup_steps > 0:
            self.warmup_factor = 1.0 / num_warmup_steps
        else:
            self.warmup_factor = 1

        self._update_learning_rate(self.warmup_factor * self.learning_rate)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return self.learning_rate

    def step(self, epoch: Optional[int] = None) -> None:
        step = epoch
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1

        if (
            self.num_warmup_steps > 0
            and self.last_step <= self.num_warmup_steps
        ):
            self.warmup_factor = self.last_step / float(self.num_warmup_steps)
            lr = self.warmup_factor * self.learning_rate
        elif self.last_step <= self.num_training_steps:
            lr_range = self.learning_rate - self.end_learning_rate
            steps_passed = self.last_step - self.num_warmup_steps
            steps_remaining = self.num_training_steps - self.num_warmup_steps
            lr = lr_range * (1 - steps_passed / steps_remaining) ** self.power
            lr += self.end_learning_rate
        else:
            lr = self.end_learning_rate

        self.learning_rate = lr
        self._update_learning_rate(lr)

    def _update_learning_rate(self, lr: float) -> None:
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
