import numpy as np

from lighter.decorator import context


class BaseCollectible:
    @context
    def __init__(self):
        self.collection = {}

    def reset(self):
        self.collection = {}

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.collection.keys():
                self.collection[key] = []
            self.collection[key].append(value)

    def redux(self, func=np.mean):
        redux = {}
        for key, value in self.collection.items():
            redux[key] = func(np.array(value))
        return redux
