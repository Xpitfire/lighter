from torch.utils.tensorboard import SummaryWriter
from lighter.decorator import context


class BaseWriter(SummaryWriter):
    """
    Base writer class descending from SummaryWriter (tensorboard) with injected 'context' instance.
    """
    @context()
    def __init__(self):
        super(BaseWriter, self).__init__()
        self.steps = 0

    def write(self, *args, **kwargs):
        pass

    def step(self):
        self.steps += 1
