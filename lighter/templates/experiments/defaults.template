from lighter.decorator import strategy
from lighter.experiment import DefaultExperiment


class Experiment(DefaultExperiment):
    @strategy(config='configs/modules.config.json')
    def __init__(self):
        super(Experiment, self).__init__()
