from lighter.decorator import strategy, device


class TrainingStrategy(object):
    @device
    @strategy
    def __init__(self):
        self.lr_scheduler = None
        self.patience = None
