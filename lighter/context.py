import petname
from threading import RLock
from lighter.config import Config
from lighter.loader import Loader
from lighter.registry import Registry
from lighter.search import ParameterSearch


class Context(object):
    """
    Application context handling all the config, registry and search parameter instances.
    """
    # mutex to protect context from multiprocess concurrent access
    _mutex = RLock()
    _instance = None

    def __init__(self, config_file: str, parse_args_override):
        """
        Create a context object and registers configs, types and instances of the defined modules.
        :param config_file: default config file
        """
        Context._mutex.acquire()
        try:
            self.registry = Registry.create_instance()
            self.config = Config.create_instance(config_file, parse_args_override)
            self.config.set_value('context_id', petname.Generate(2, '-', 6))
            self.search = ParameterSearch.create_instance()
            self.instantiate_types(self.registry.types)
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
    def create(config_file: str = None, parse_args_override: bool = True) -> "Context":
        """
        Create application context threadsafe.
        :param config_file: Initial config file for context to load.
        :param parse_args_override: Allows to override configs according to the command line arguments.
        :return:
        """
        Context._mutex.acquire()
        try:
            Context._instance = Context(config_file, parse_args_override)
            return Context._instance
        finally:
            Context._mutex.release()
