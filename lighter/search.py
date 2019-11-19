from typing import List
from lighter.config import Config
from threading import RLock
from lighter.misc import DotDict
from lighter.parameter import Parameter


class ParameterSearch(DotDict):
    """
    Parameter search collection updated by the @search decorator.
    """
    _mutex = RLock()
    _instance = None

    def __init__(self, **kwargs):
        super(ParameterSearch, self).__init__(**kwargs)

    @staticmethod
    def compile(params: List["Parameter"]) -> List[Config]:
        configs = []
        for i, param in enumerate(params):
            configs = [c for c in param]
            rest = [p for j, p in enumerate(params) if i != j]
            for c in configs:
                [p.update_config(c) for p in rest]
                configs = configs + ParameterSearch.compile(rest)
        return configs

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
