from threading import RLock
from box import Box


class Registry(object):
    """
    Instance and type registry for properties loaded from configs.
    """
    _instance = None
    # mutex for threadsafe usage
    _mutex = RLock()

    def __init__(self, **kwargs):
        """
        Creates a new Registry object storing and handling all types and instances imported
        from configs.
        :param kwargs:
        """
        super(Registry, self).__init__(**kwargs)
        self.instances = Box()
        self.types = Box()

    def register_type(self, name, type_):
        setattr(self.types, name, type_)

    def register_instance(self, name, instance):
        setattr(self.instances, name, instance)

    def unregister_type(self, name):
        del self.types[name]

    def unregister_instance(self, name):
        del self.instances[name]

    def contains_type(self, name):
        return name in self.types.keys()

    def contains_instance(self, name):
        return name in self.instances.keys()

    @staticmethod
    def get_instance() -> "Registry":
        Registry._mutex.acquire()
        try:
            return Registry._instance
        finally:
            Registry._mutex.release()

    @staticmethod
    def create_instance() -> "Registry":
        Registry._mutex.acquire()
        try:
            Registry._instance = Registry()
            return Registry._instance
        finally:
            Registry._mutex.release()
