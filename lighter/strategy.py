from lighter.decorator import device, context


class BaseTrainingStrategy(object):
    """
    Base class for defining a training strategy.
    """
    @device
    @context
    def __init__(self):
        pass
