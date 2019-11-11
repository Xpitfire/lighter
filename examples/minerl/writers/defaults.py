from lighter.decorator import config
from lighter.writer import BaseWriter


class SimpleWriter(BaseWriter):
    @config(path='minerl/writers/defaults.config.json', group='writer')
    def __init__(self):
        super(SimpleWriter, self).__init__()

    def write(self, category, *args, **kwargs):
        if self.config.writer.enable_logs:
            for key, value in kwargs.items():
                self.add_scalar('{}/{}'.format(category, key), value, self.steps)