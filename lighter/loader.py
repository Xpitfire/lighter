import importlib
import traceback
import logging

from threading import RLock
from lighter.exceptions import MultipleInstanceError
from lighter.registry import Registry


class Loader(object):
    _instance = None
    # mutex for threadsafe instance loading
    _mutex = RLock()

    def __init__(self):
        """
        Creates a loader object which registers new modules.
        """
        self.registry = Registry.get_instance()
        if Loader._instance is None:
            Loader._instance = self
        else:
            raise MultipleInstanceError()

    def import_modules(self, modules: dict):
        """
        Registers a dictionary of modules to the registry.
        :param modules: list of types
        :return:
        """
        for k, v in modules.items():
            if isinstance(v, str):
                module = Loader.import_path(v)
                setattr(self.registry.types, k, module)

    @staticmethod
    def import_path(name: str):
        """
        Imports a defined python type.
        :param name: path to a python module
        :return:
        """
        type_ = None
        try:
            components = name.split('.')
            module = importlib.import_module('.'.join(components[:-1]))
            type_ = getattr(module, components[-1])
        except Exception as e:
            tb = traceback.format_exc()
            logging.warning('Loader: Could not import: {} - {}'.format(name, e))
            logging.warning(tb)
        return type_

    @staticmethod
    def get_instance() -> "Loader":
        Loader._mutex.acquire()
        try:
            if Loader._instance is None:
                Loader._instance = Loader()
            return Loader._instance
        finally:
            Loader._mutex.release()
