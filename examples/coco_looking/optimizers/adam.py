import torch

from lighter.decorator import config, model
from lighter.optimizer import BaseOptimizer


class Optimizer(BaseOptimizer):
    @model
    @config(path='optimizers/defaults.config.json', property='optimizer')
    def __init__(self):
        super(Optimizer, self).__init__()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.optimizer.lr
        )
