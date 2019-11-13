from typing import Callable
from lighter.config import Config


class Parameter(object):
    """
    Base Parameter class for handling hyperparameter search.
    """
    def __init__(self, ref):
        super(Parameter, self).__init__()
        self.config = Config.get_instance()
        self.ref = ref

    @property
    def value(self):
        return self.config.get_value(self.ref)

    @value.setter
    def value(self, value):
        self.config.set_value(self.ref, value)


class GridParameter(Parameter):
    """
    Grid search parameter with linear unified step sizes.
    """
    def __init__(self, ref: str, min: float, max: float, step: float = 1.0):
        super(GridParameter, self).__init__(ref)
        self.min = min
        self.max = max
        self.step = step

    def __iter__(self):
        self.value = self.min
        return self

    def __next__(self):
        if self.value < self.max:
            value = self.value
            self.value += self.step
            return value
        else:
            raise StopIteration


class CallableGridParameter(Parameter):
    """
    Grid search parameter with lambda function based step sizes which allows non linear changes.
    """
    def __init__(self, ref: str, min: float, max: float, step: Callable[[float], float]):
        super(CallableGridParameter, self).__init__(ref)
        self.min = min
        self.max = max
        self.step = step

    def __iter__(self):
        self.value = self.min
        return self

    def __next__(self):
        if self.value < self.max:
            value = self.value
            self.value = self.step(self.value)
            return value
        else:
            raise StopIteration


class AnnealParameter(Parameter):
    """
    Annealing search parameter which allows lambda function non-linear decline.
    """
    def __init__(self, ref: str, start: float, threshold: float, anneal: Callable[[float], float]):
        super(AnnealParameter, self).__init__(ref)
        self.start = start
        self.threshold = threshold
        self.anneal = anneal

    def __iter__(self):
        self.value = self.start
        return self

    def __next__(self):
        if self.value > self.threshold:
            value = self.value
            self.value = self.anneal(self.value)
            return value
        else:
            raise StopIteration
