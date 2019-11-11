import torch

from lighter.decorator import context


class BaseCriterion(torch.nn.Module):
    """
    Criterion base class with injected application 'context'.
    """
    @context
    def __init__(self):
        super(BaseCriterion, self).__init__()
