from threading import RLock

from lighter.config import Config
from lighter.registry import Registry


class Context(object):
    _mutex = RLock()
    _instance = None

    def __init__(self, config_file: str):
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
        return Context._instance

    @staticmethod
    def create(config_file: str = None):
        Context._mutex.acquire()
        try:
            if Context._instance is None:
                Context._instance = Context(config_file)
        finally:
            Context._mutex.release()
