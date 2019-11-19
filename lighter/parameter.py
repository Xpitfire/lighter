import os
from typing import Callable, List
from lighter.config import Config
from lighter.misc import DotDict


class Parameter(object):
    """
    Base Parameter class for handling hyperparameter search.
    """
    def __init__(self, ref):
        super(Parameter, self).__init__()
        config = Config.get_instance()
        if config is None:
            config = Config()
        self.config = config.copy()
        self.ref = ref
        self.compilation: List[Config] = []

    def __iter__(self):
        raise NotImplementedError('Parameter: No implementation found!')

    def __next__(self):
        raise NotImplementedError('Parameter: No implementation found!')

    def list_values(self):
        raise NotImplementedError('Parameter: No implementation found!')

    @property
    def value(self):
        return self.config.get_value(self.ref)

    @value.setter
    def value(self, value):
        self.config.set_value(self.ref, value)

    def update_config(self, config):
        self.config = config.copy()


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
            config = self.config.copy()
            self.value += self.step
            return config
        else:
            raise StopIteration

    def list_values(self):
        return list([self.min + i * self.step for i in range(int((self.max - self.min) // self.step))])


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
            config = self.config.copy()
            self.value = self.step(self.value)
            return config
        else:
            raise StopIteration

    def list_values(self):
        val = self.min
        values = [val]
        while val < self.max:
            val = self.step(val)
            values.append(val)
        return values


class AnnealParameter(Parameter):
    """
    Annealing search parameter which allows lambda function non-linear decline.
    """
    def __init__(self, ref: str, start: float, threshold: float, anneal: Callable[[float], float]):
        super(AnnealParameter, self).__init__(ref)
        self.start = start
        self.threshold = threshold
        assert self.start >= self.threshold
        self.anneal = anneal

    def __iter__(self):
        self.value = self.start
        return self

    def __next__(self):
        if self.value >= self.threshold:
            config = self.config.copy()
            self.value = self.anneal(self.value)
            return config
        else:
            raise StopIteration

    def list_values(self):
        val = self.start
        values = [val]
        while val >= self.threshold:
            val = self.anneal(val)
            values.append(val)
        return values


class BinaryParameter(Parameter):
    """
    Binary search parameter which returns true and false each once.
    """
    def __init__(self, ref: str):
        super(BinaryParameter, self).__init__(ref)

    def __iter__(self):
        self.idx = 0
        self.value = False
        return self

    def __next__(self):
        if self.idx < 2:
            config = self.config.copy()
            self.value = True
            self.idx += 1
            return config
        else:
            raise StopIteration

    def list_values(self):
        return [True, False]


class StrategyParameter(Parameter):
    """
    Strategy parameters allow to iterate over different training strategies.
    """
    def __init__(self, ref: str, options: List[str]):
        super(StrategyParameter, self).__init__(ref)
        self.options = options

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < len(self.options):
            config = self.config.copy()
            imported_config, _ = Config.load(path=os.path.join(self.ref, self.options[self.idx]))
            for k, v in imported_config.items():
                setattr(self.config, k, v)
            self.idx += 1
            return config
        else:
            raise StopIteration

    def list_values(self):
        return self.options
