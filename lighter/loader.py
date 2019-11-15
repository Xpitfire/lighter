import importlib
import traceback
import logging

from threading import RLock
from lighter.registry import Registry


class Loader(object):
    _instance = None
    # mutex for threadsafe instance loading
    _mutex = RLock()

    @staticmethod
    def import_modules(modules: dict):
        """
        Registers a dictionary of modules to the registry.
        :param modules: list of types
        :return:
        """
        for k, v in modules.items():
            if isinstance(v, str):
                module = Loader.import_path(v)
                setattr(Registry.get_instance().types, k, module)

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
