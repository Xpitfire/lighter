from threading import RLock

import importlib
import traceback
import logging

from lighter.exceptions import MultipleInstanceError
from lighter.registry import Registry


class Loader(object):
    _instance = None
    _mutex = RLock()

    def __init__(self):
        self.registry = Registry.get_instance()
        if Loader._instance is None:
            Loader._instance = self
        else:
            raise MultipleInstanceError()

    def import_modules(self, modules):
        for k, v in modules.items():
            if isinstance(v, str) and 'class::' in v:
                module = Loader.import_path(v)
                setattr(self.registry.types, k, module)

    @staticmethod
    def import_path(name: str):
        class_ = None
        try:
            components = name.split('.')
            module = importlib.import_module('.'.join(components[:-1]))
            class_ = getattr(module, components[-1])
        except Exception as e:
            tb = traceback.format_exc()
            logging.warning('Loader: Could not import: {} - {}'.format(name, e))
            logging.warning(tb)
        return class_

    @staticmethod
    def get_instance():
        Loader._mutex.acquire()
        try:
            if Loader._instance is None:
                Loader._instance = Loader()
            return Loader._instance
        finally:
            Loader._mutex.release()
