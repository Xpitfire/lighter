from lighter.decorator import strategy
from lighter.experiment import DefaultExperiment


class Experiment(DefaultExperiment):
    @strategy(config='configs/coco_looking.config.json')
    def __init__(self):
        super(Experiment, self).__init__(epochs=50)
