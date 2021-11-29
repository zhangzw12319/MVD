"""
Learning rate scheduler
"""
import mindspore as ms

from mindspore import ops as P
from mindspore import nn

class LRScheduler(nn.Cell):
    r"""
    Gets learning rate warming up + decay.

    Args:
        learning_rate (float): The initial value of learning rate.
        warmup_steps (int): The warm up steps of learning rate.
        weight_decay (int): The weight decay steps of learning rate.

    Inputs:
        Tensor. The current step number.

    Outputs:
        Tensor. The learning rate value for the current step.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self, learning_rate, warmup_steps=0, weight_decay=None, decay_factor=0.5):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.decay_factor = decay_factor
        self.min = P.Minimum()
        self.cast = P.Cast()

    def construct(self, global_step, **kwargs):
        """
        LR Scheduler
        """
        if global_step < self.warmup_steps:
            warmup_percent = self.cast(self.min(global_step, self.warmup_steps), ms.float32)\
             / self.warmup_steps
            return self.learning_rate * warmup_percent
        lr_ = self.learning_rate
        for decay in self.weight_decay:
            if global_step <= decay:
                break
            lr_ = lr_ * self.decay_factor
        lr_ = self.cast(lr_, ms.float32)
        return lr_
