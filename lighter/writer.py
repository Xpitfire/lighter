import socket
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from lighter.decorator import context


class BaseWriter(SummaryWriter):
    """
    Base writer class descending from SummaryWriter (tensorboard) with injected 'context' instance.
    """
    @context
    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix=''):
        if log_dir is None:
            timestamp = datetime.timestamp(datetime.now())
            log_dir = 'runs/{}-{}-{}'.format(timestamp, self.config.context_id, socket.gethostname())
        super(BaseWriter, self).__init__(log_dir=log_dir, comment=comment,
                                         purge_step=purge_step, max_queue=max_queue,
                                         flush_secs=flush_secs, filename_suffix=filename_suffix)
        self.steps = 0

    def write(self, category: str = None, *args, **kwargs):
        for key, value in kwargs.items():
            # remove collectible key prefix if available
            val = key.split('$_')
            cat, key = val[0], val[1]
            if category is not None and cat != "${}".format(category):
                continue
            # prepend category if available
            if category is not None:
                key = '{}/{}'.format(category, key)
            self.add_scalar(key, value, self.steps)

    def step(self):
        self.steps += 1
