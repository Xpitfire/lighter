from typing import List
from box import Box
from lighter.config import Config
from threading import RLock
from lighter.misc import generate_long_id
from lighter.parameter import Parameter


class ParameterSearch(Box):
    """
    Parameter search collection updated by the @search decorator.
    """
    _mutex = RLock()
    _instance = None

    def __init__(self, **kwargs):
        super(ParameterSearch, self).__init__(**kwargs)

    @staticmethod
    def compile(params: List["Parameter"], config: Config = None) -> List[Config]:
        configs = []
        if len(params) <= 0:
            # fix to get unique name
            config['experiment_id'] = generate_long_id()
            return [config]
        param = params[0]
        if config is not None:
            param.update_config(config)
        for p in param:
            configs = configs + ParameterSearch.compile(params[1:], p)
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
