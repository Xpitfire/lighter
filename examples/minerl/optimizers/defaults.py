import torch

from lighter.decorator import config
from lighter.optimizer import BaseOptimizer


class SimpleOptimizer(BaseOptimizer):
    @config(path='minerl/optimizers/defaults.config.json', group='optimizer')
    def __init__(self):
        super(SimpleOptimizer, self).__init__()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.optimizer.lr,
            momentum=self.config.optimizer.momentum,
            weight_decay=self.config.optimizer.weight_decay
        )
