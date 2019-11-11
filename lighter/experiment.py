from lighter.decorator import experiment, device


class BaseExperiment(object):
    """
    Experiment base class to create new algorithm runs.
    """
    @device()
    @experiment()
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)

    def eval(self):
        """
        Evaluates an experiment.
        :return:
        """
        raise NotImplementedError('BaseExperiment: No implementation found!')

    def train(self):
        """
        Training instance for the experiment run.
        :return:
        """
        raise NotImplementedError('BaseExperiment: No implementation found!')

    def run(self, *args, **kwargs):
        """
        Main run method usually containing the training loop.
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError('BaseExperiment: No implementation found!')
