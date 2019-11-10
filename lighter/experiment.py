from lighter.decorator import experiment


class BaseExperiment(object):
    """
    Experiment base class to create new algorithm runs.
    """
    @experiment
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)

    def eval(self):
        raise NotImplementedError('BaseExperiment: No implementation found!')

    def train(self):
        raise NotImplementedError('BaseExperiment: No implementation found!')

    def run(self, *args, **kwargs):
        raise NotImplementedError('BaseExperiment: No implementation found!')
