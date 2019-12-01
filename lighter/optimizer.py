from lighter.decorator import context


class BaseOptimizer(object):
    """
    Base class proxy for the PyTorch optimizers with injected 'model' and 'context' instances.
    """
    @context
    def __init__(self):
        self.optimizer = None

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.state_dict(state_dict)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, closure=None):
        self.optimizer.step(closure)

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)
