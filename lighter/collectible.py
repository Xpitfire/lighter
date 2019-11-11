import numpy as np
from lighter.config import Config

from lighter.decorator import context
from lighter.misc import DotDict


class BaseCollectible:
    """
    Base class for collectibles. They are used to store and update values over multiple steps / epochs.
    Collectibles are passed a dictionary containing key-value pairs whereas the key
    determines the collection for adding values.
    """
    @context
    def __init__(self):
        self.collection = DotDict()

    def reset(self):
        """
        Resets the collectible instance.
        :return:
        """
        self.collection = DotDict()

    def update(self, **kwargs):
        """
        Updates a collectible key-value pair.
        Dict: {
            'name': <number>,
            ...
        }
        :param kwargs: key-value data pair.
        :return:
        """
        for key, value in kwargs.items():
            if key not in self.collection.keys():
                self.collection[key] = []
            self.collection[key].append(value)

    def redux(self, func=np.mean):
        """
        Reduces the collected values to a single number.
        :param func: Reduction function applied on the collected values.
        :return: Dictionary with reduced single numbers per key.
        """
        redux = {}
        for key, value in self.collection.items():
            redux[key] = func(np.array(value))
        return redux
