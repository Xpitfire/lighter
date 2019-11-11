from threading import RLock

from lighter.config import Config
from lighter.registry import Registry


class Context(object):
    # mutex to protect context from multiprocess concurrent access
    _mutex = RLock()
    _instance = None

    def __init__(self, config_file: str):
        """
        Create a context object and registers configs, types and instances of the defined modules.
        :param config_file: default config file
        """
        Context._mutex.acquire()
        try:
            self.config = Config.create_instance(config_file)
            self.registry = Registry.get_instance()
            self._instantiate_types()
        finally:
            Context._mutex.release()

    def _instantiate_types(self):
        for name, class_ in self.registry.types.items():
            self.registry.register_instance(name, class_())

    @staticmethod
    def get_instance():
        Context._mutex.acquire()
        try:
            return Context._instance
        finally:
            Context._mutex.release()

    @staticmethod
    def create(config_file: str = None):
        """
        Create application context threadsafe.
        :param config_file:
        :return:
        """
        Context._mutex.acquire()
        try:
            if Context._instance is None:
                Context._instance = Context(config_file)
        finally:
            Context._mutex.release()
