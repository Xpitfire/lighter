import torch

from lighter.decorator import config
from lighter.optimizer import BaseOptimizer


class Optimizer(BaseOptimizer):
    @config(path='examples/coco_looking/optimizers/defaults.config.json', group='optimizer')
    def __init__(self):
        super(Optimizer, self).__init__()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.optimizer.lr,
            momentum=self.config.optimizer.momentum,
            weight_decay=self.config.optimizer.weight_decay
        )
