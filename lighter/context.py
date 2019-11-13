from threading import RLock

from lighter.config import Config
from lighter.registry import Registry
from lighter.search import ParameterSearch


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
            self.instantiate_types(self.registry.types)
            self.search = ParameterSearch.get_instance()
        finally:
            Context._mutex.release()

    def instantiate_types(self, types):
        Context._mutex.acquire()
        try:
            for name, class_ in types.items():
                self.registry.register_instance(name, class_())
        finally:
            Context._mutex.release()

    @staticmethod
    def get_instance() -> "Context":
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
