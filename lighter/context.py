import traceback
from threading import RLock

import logging

from lighter.config import Config
from lighter.exceptions import TypeInstantiationError
from lighter.misc import generate_short_id
from lighter.registry import Registry
from lighter.search import ParameterSearch

import numpy as np

# TODO: remove in future terms
import warnings
warnings.simplefilter('ignore', FutureWarning)


class Context(object):
    """
    Application context handling all the config, registry and search parameter instances.
    """
    # mutex to protect context from multiprocess concurrent access
    _mutex = RLock()
    _instance = None

    def __init__(self,
                 config_file: str,
                 config_dict: dict,
                 parse_args_override,
                 device: str = None,
                 auto_instantiate_types: bool = True,
                 allow_context_changes: bool = True):
        """
        Create a context object and registers configs, types and instances of the defined modules.
        :param config_file: default config file
        """
        Context._mutex.acquire()
        try:
            self.registry = Registry.create_instance()
            self.config = Config.create_instance(config_file,
                                                 config_dict=config_dict,
                                                 parse_args_override=parse_args_override,
                                                 device=device)
            self.config.set_value('context_id', generate_short_id())
            self.search = ParameterSearch.create_instance()
            self.allow_context_changes = allow_context_changes
            self.auto_instantiate_types = auto_instantiate_types
            if auto_instantiate_types:
                # if auto instantiation of types is enables, and no context is available this requires to assign
                # the current instance to the global instance
                if Context._instance is None:
                    Context._instance = self
                self.instantiate_types()
        finally:
            Context._mutex.release()

    def _resolve_dependency_graph(self, types: dict, num_retries: int = 0, max_retries: int = np.inf, error=[]):
        """
        Registers instances with retries if graph resolving fails due to transitive dependencies between
        modules.
        TODO: improve by different prioritization technique
        :param types: types to instantiate
        :param num_retries: current retry count
        :param max_retries: maximum recursion depth
        :return:
        """
        retry = {}
        for name, class_ in types.items():
            try:
                self.registry.register_instance(name, class_())
            except Exception:
                err = traceback.format_exc()
                error.append(err)
                retry[name] = class_
        if len(retry) > 0 and num_retries < max_retries:
            num_retries += 1
            self._resolve_dependency_graph(retry, num_retries, max_retries)
        if num_retries >= max_retries:
            for e in error:
                logging.exception(e, exc_info=True)
            raise TypeInstantiationError('Could not resolve the dependency graph due to type creation errors.')

    def instantiate_types(self, types: dict = None):
        """
        Instantiates and registers types to the registry instances.
        :param types: types to instantiate and register
        :return:
        """
        Context._mutex.acquire()
        if types is None:
            types = self.registry.types
        try:
            len_types = len(list(types.keys()))
            if len_types > 0:
                self._resolve_dependency_graph(types, 0, max_retries=len_types)
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
    def create(config_file: str = None,
               config_dict: dict = None,
               parse_args_override: bool = True,
               device: str = None,
               auto_instantiate_types: bool = True,
               allow_context_changes: bool = True) -> "Context":
        """
        Create application context threadsafe.
        :param config_file: Initial config file for context to load.
        :param config_dict: Initial config json for context to load.
        :param parse_args_override: Allows to override configs according to the command line arguments.
        :param auto_instantiate_types: Instantiates types after been registered
        :param device: specifies the running device
        :param allow_context_changes: If this is set to False, it prevents all decorators from modifying updates
               on the configs, context and registry
        :return:
        """
        Context._mutex.acquire()
        try:
            Context._instance = Context(config_file,
                                        config_dict=config_dict,
                                        parse_args_override=parse_args_override,
                                        auto_instantiate_types=auto_instantiate_types,
                                        device=device,
                                        allow_context_changes=allow_context_changes)
            return Context._instance
        finally:
            Context._mutex.release()
