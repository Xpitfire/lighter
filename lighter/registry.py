from threading import RLock

from lighter.misc import DotDict


class Registry(DotDict):
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
        self.instances = DotDict()
        self.types = DotDict()

    def register_type(self, name, type_):
        setattr(self.types, name, type_)

    def register_instance(self, name, instance):
        setattr(self.instances, name, instance)

    def unregister_type(self, name):
        del self.types[name]

    def unregister_instance(self, name):
        del self.instances[name]

    @staticmethod
    def get_instance():
        Registry._mutex.acquire()
        try:
            if Registry._instance is None:
                Registry._instance = Registry()
            return Registry._instance
        finally:
            Registry._mutex.release()
