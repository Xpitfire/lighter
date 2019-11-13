from lighter.decorator import device, context


class BaseTrainingStrategy(object):
    @device
    @context
    def __init__(self):
        pass
