from lighter.decorator import strategy, references
from lighter.experiment import DefaultExperiment


class Experiment(DefaultExperiment):
    @strategy(config='configs/coco_looking.config.json')
    @references
    def __init__(self):
        super(Experiment, self).__init__(epochs=50)
