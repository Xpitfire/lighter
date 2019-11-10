import torch

from lighter.decorator import context


class BaseModule(torch.nn.Module):
    @context
    def __init__(self):
        super(BaseModule, self).__init__()
