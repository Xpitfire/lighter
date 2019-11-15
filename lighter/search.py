from threading import RLock
from lighter.misc import DotDict


class ParameterSearch(DotDict):
    """
    Parameter search collection updated by the @search decorator.
    """
    _mutex = RLock()
    _instance = None

    def __init__(self, **kwargs):
        super(ParameterSearch, self).__init__(**kwargs)

    @staticmethod
    def get_instance() -> "ParameterSearch":
        ParameterSearch._mutex.acquire()
        try:
            return ParameterSearch._instance
        finally:
            ParameterSearch._mutex.release()

    @staticmethod
    def create_instance() -> "ParameterSearch":
        ParameterSearch._mutex.acquire()
        try:
            ParameterSearch._instance = ParameterSearch()
            return ParameterSearch._instance
        finally:
            ParameterSearch._mutex.release()
