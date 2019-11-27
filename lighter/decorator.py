import functools
import logging
import inspect
from typing import Tuple, List, Callable
from enum import Enum
from box import Box
from lighter.config import Config
from lighter.context import Context
from lighter.exceptions import DependencyInjectionError
from lighter.loader import Loader
from lighter.misc import DotDict
from lighter.registry import Registry
from lighter.search import ParameterSearch
from lighter.parameter import Parameter

DEFAULT_PROPERTIES = ['transform', 'dataset', 'data_builder', 'model', 'optimizer',
                      'collectible', 'criterion', 'metric', 'writer']


def _handle_injections(args, injectables: List[str], ignore_none_values):
    registry = Registry.get_instance()
    instance = args[0]

    for inject in injectables:
        parent, name = DotDict.resolve(registry.instances, inject)
        value = getattr(parent, name, None)
        if not ignore_none_values and value is None:
            raise DependencyInjectionError('Trying to inject dependency of unresolved '
                                           'key: {} into instance: {} with value: {}'
                                           .format(inject, instance, value))
        setattr(instance, name, value)


def _handle_config(args, path, source):
    context = Context.get_instance()
    config = Config.get_instance()
    instance = args[0]
    # check if context is allowed to change and perform config updates
    if context.allow_context_changes and path is not None:
        imported_config, _ = Config.load(path=path)
        if source is not None:
            parent, name = DotDict.resolve(config, source)
            setattr(parent, name, imported_config)
        else:
            for k, v in imported_config.items():
                setattr(config, k, v)
    setattr(instance, 'config', config)


def _handle_registration(source):
    context = Context.get_instance()
    # prevent type registration is context is not allowed to change
    if not context.allow_context_changes:
        return
    config = Config.get_instance()
    parent, name = DotDict.resolve(config, source)
    types = getattr(parent, name)
    types = {k: Loader.import_path(v[len("type::"):]) for k, v in types.strategy.items()}
    context.instantiate_types(types)


def _wrapper_delegate(func, injectables, ignore_none_values):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _handle_injections(args, injectables, ignore_none_values)
        return func(*args, **kwargs)
    return wrapper


def _search_and_load_type(instance):
    registry = Registry.get_instance()
    reg_instance, key = DotDict.resolve(registry.types, instance)
    class_ = getattr(reg_instance, key, None)
    if class_ is None or not inspect.isclass(class_):
        raise DependencyInjectionError()
    new_ = class_()
    registry.register_instance(key, new_)
    return new_


# ----------- PUBLIC DECORATORS -----------


def device(func=None, name: str = None, source: str = 'device.default', property: str = 'device'):
    """
    Injects a device info into the current object instance.
    :param func: original function instance
    :param name: device id if pre-specified
    :param source: source path for the config
    :param property: device name for injection
    :return:
    """
    if not func:
        # workaround to enable empty decorator
        return functools.partial(device, name=name, source=source, property=property)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        context = Context.get_instance()
        config = Config.get_instance()
        obj_instance = args[0]

        # update device according to id and check if context is allowed to be updated
        if context.allow_context_changes and name is not None:
            config.set_value(source, name)

        value = config.get_value(source)
        # if never set but called, then through exception
        if value is None:
            logging.error('Trying to inject dependency of unresolved name: {}'.format(property))
            raise DependencyInjectionError()

        setattr(obj_instance, property, value)
        return func(*args, **kwargs)
    return wrapper


def config(func=None, path: str = None, property: str = None):
    """
    Config decorator to import and inject the specified config.
    This config instance can be grouped in a defined name.
    The default config containing this new property is injected into an object instance.
    If none of the options is set, then only the default config if injected.
    :param func: original function instance
    :param path: path of the config file
    :param property: source name to categorize new config instance
    :return:
    """
    if not func:
        # workaround to enable empty decorator
        return functools.partial(config, path=path, property=property)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _handle_config(args, path, property)
        return func(*args, **kwargs)
    return wrapper


def context(func):
    """
    Context decorator to inject the default context instance.
    :param func: original function do inject instances
    :return:
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        context_ = Context.get_instance()
        config = Config.get_instance()
        registry = Registry.get_instance()
        search = ParameterSearch.get_instance()
        instance = args[0]

        setattr(instance, 'context', context_)
        setattr(instance, 'config', config)
        setattr(instance, 'registry', registry)
        setattr(instance, 'search', search)
        return func(*args, **kwargs)
    return wrapper


class InjectOption(Enum):
    Type = 0
    Instance = 1
    Config = 2
    Search = 3


def inject(source: str, property: str, option: InjectOption = InjectOption.Instance):
    """
    Injects a named variable of any given instance into the decorated object instance.
    :param source: Source instance available in the registry.
    :param property: Target name of the variable to inject.
    :param option: Option for the registry lookup.
    :return:
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            context = Context.get_instance()
            obj_instance = args[0]

            # if instances option selected lookup attribute or load if not already instantiated
            if option == InjectOption.Instance:
                registry = Registry.get_instance()
                instance, key = DotDict.resolve(registry.instances, source)
                value = getattr(instance, key, None)
                # prevent all side effects if allow_context_changes is False
                if context.allow_context_changes and value is None:
                    value = _search_and_load_type(source)
            # otherwise if types then return simple type
            elif option == InjectOption.Type:
                registry = Registry.get_instance()
                instance, key = DotDict.resolve(registry.types, source)
                value = getattr(instance, key, None)
            elif option == InjectOption.Config:
                config = Config.get_instance()
                instance, key = DotDict.resolve(config, source)
                value = getattr(instance, key, None)
            elif option == InjectOption.Search:
                search = ParameterSearch.get_instance()
                instance, key = DotDict.resolve(search, source)
                value = getattr(instance, key, None)
            else:
                raise NotImplementedError()

            # if value is still None then through error, since it was never registered or failed to load
            if value is None:
                logging.error('Trying to inject dependency of unresolved name: {}'.format(property))
                raise DependencyInjectionError()

            setattr(obj_instance, property, value)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def register(type: str, property: str = None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            context = Context.get_instance()
            # disable register to prevent all side effects on the registry
            if not context.allow_context_changes:
                return func(*args, **kwargs)

            obj_instance = args[0]

            registry = Registry.get_instance()

            instance, key = DotDict.resolve(registry.instances, type)
            setattr(instance, key, type)

            registry.register_type(key, type)
            module = Loader.import_path(type)
            value = module()
            registry.register_instance(key, value)

            if property is not None:
                setattr(obj_instance, property, value)

            return func(*args, **kwargs)
        return wrapper
    return decorator


def hook(method: str, replace_with: Callable, *args, **kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args_, **kwargs_):
            result = func(*args_, **kwargs_)
            instance = args_[0]
            m = getattr(instance, method)
            setattr(instance, method, lambda: replace_with(m, *args, **kwargs))
            return result
        return wrapper
    return decorator


def strategy(config: str, source: str = 'strategy', properties: list = None, ignore_none_values: bool = True):
    """
    Strategy decorator to inject a training strategy instance.
    :param source: defines the source name of the current training strategy
    :param config: property file for the current training strategy
    :param properties: names of the injection variable as specified in the config
    :param ignore_none_values: ignores None value references of injections
    :return:
    """
    if properties is None:
        properties = DEFAULT_PROPERTIES

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _handle_config(args, config, source)
            _handle_registration(source)
            _handle_injections(args, properties, ignore_none_values=ignore_none_values)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def reference(name: str):
    """
    Reference decorator to inject multiple values at once without registering new objects.
    :param name: Property to inject.
    :return:
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _handle_injections(args, [name], ignore_none_values=False)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def references(func=None, names: list = None, ignore_none_values: bool = True):
    """
    Reference decorator to inject multiple values at once without registering new objects.
    If no list is provided the default properties will be injected:
    'transform', 'dataset', 'data_builder', 'model', 'optimizer', 'collectible', 'criterion', 'metric', 'writer'
    :param func: original function instance
    :param names: List of properties to inject.
    :param ignore_none_values: ignores None value references of injections
    :return:
    """
    if names is None:
        names = DEFAULT_PROPERTIES
    if not func:
        # workaround to enable empty decorator
        return functools.partial(references, names=names, ignore_none_values=ignore_none_values)
    return _wrapper_delegate(func, names, ignore_none_values=ignore_none_values)


def transform(func):
    """
    Transform decorator to inject the default transform instance.
    :param func: original function instance
    :return:
    """
    properties = ['transform']
    if not func:
        # workaround to enable empty decorator
        return functools.partial(transform)
    return _wrapper_delegate(func, properties, ignore_none_values=False)


def dataset(func):
    """
    Dataset decorator to inject the default dataset instance.
    :param func: original function instance
    :return:
    """
    properties = ['dataset']
    if not func:
        # workaround to enable empty decorator
        return functools.partial(dataset)
    return _wrapper_delegate(func, properties, ignore_none_values=False)


def model(func):
    """
    Model decorator to inject the default model instance.
    :param func: original function instance
    :return:
    """
    properties = ['model']
    if not func:
        # workaround to enable empty decorator
        return functools.partial(model)
    return _wrapper_delegate(func, properties, ignore_none_values=False)


def metric(func):
    """
    Metric decorator to inject the default metric instance.
    :param func: original function instance
    :return:
    """
    properties = ['metric']
    if not func:
        # workaround to enable empty decorator
        return functools.partial(metric)
    return _wrapper_delegate(func, properties, ignore_none_values=False)


def search(group: str = None, params: List[Tuple[str, Parameter]] = None):
    """
    Decorator for searching hyper-parameters.
    :param group: collects the params to search only in an grouped logical search space
    :param params: list of tuples containing searchable parameters
    :return:
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            search = ParameterSearch.get_instance()
            context = Context.get_instance()
            instance = args[0]

            setattr(instance, 'search', search)
            # prevent all side effects on search
            if not context.allow_context_changes:
                return func(*args, **kwargs)

            if group is None and params is None:
                return func(*args, **kwargs)

            if group is None:
                property = search
            else:
                if group not in search.keys():
                    property = Box()
                    search[group] = property
                property = search[group]

            if params is not None:
                for param in params:
                    property[param[0]] = param[1]
            return func(*args, **kwargs)
        return wrapper
    return decorator
