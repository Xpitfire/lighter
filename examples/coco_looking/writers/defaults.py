import os
from lighter.decorator import config
from lighter.writer import BaseWriter


class Writer(BaseWriter):
    @config(path='writers/defaults.config.json', property='writer')
    def __init__(self):
        super(Writer, self).__init__(log_dir=os.path.join(self.config.writer.log_dir,
                                                          self.config.experiment_name))

    def write(self, category, *args, **kwargs):
        if self.config.writer.enable_logs:
            for key, value in kwargs.items():
                self.add_scalar('{}/{}'.format(category, key), value, self.steps)
