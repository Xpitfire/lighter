import functools

import inspect
import torch

from lighter.config import Config
from lighter.exceptions import DependencyInjectionError
from lighter.misc import DotDict
from lighter.registry import Registry, RegistryOption


def wrapper_delegate(func, injectables):
    """
    Delegate wrapper to inject values.
    :param func: original decorated function
    :param injectables: name of the value to inject
    :return:
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        config = Config.get_instance()
        registry = Registry.get_instance()
        instance = args[0]

        setattr(instance, 'config', config)
        setattr(instance, 'registry', registry)

        for name in injectables:
            value = getattr(registry.instances, name, None)
            if value is None:
                raise DependencyInjectionError()
            setattr(instance, name, value)

        return func(*args, **kwargs)
    return wrapper


def experiment(func=None, injectables: list = None):
    """
    Experiment decorator to inject multiple values at once.
    :param func: original function instance
    :param injectables: list of values to inject
    :return:
    """
    if injectables is None:
        injectables = ['model', 'dataset', 'data_builder', 'optimizer',
                       'collectible', 'criterion', 'metric', 'writer']

    if not func:
        # workaround to enable empty decorator
        return functools.partial(experiment, injectables=injectables)

    return wrapper_delegate(func, injectables)


def dataset(func=None, names: list = None):
    """
    Dataset decorator to inject the default dataset instance.
    :param func: original function instance
    :param names: names of the injection variable as specified in the config
    :return:
    """
    if names is None:
        names = ['dataset']

    if not func:
        # workaround to enable empty decorator
        return functools.partial(dataset, names=names)

    return wrapper_delegate(func, names)


def model(func=None, names: list = None):
    """
    Model decorator to inject the default model instance.
    :param func: original function instance
    :param names: names of the injection variable as specified in the config
    :return:
    """
    if names is None:
        names = ['model']

    if not func:
        # workaround to enable empty decorator
        return functools.partial(model, names=names)

    return wrapper_delegate(func, names)


def strategy(func=None, names: list = None):
    """
    Strategy decorator to inject the default training strategy instance.
    :param func: original function instance
    :param names: names of the injection variable as specified in the config
    :return:
    """
    if names is None:
        names = ['strategy']

    if not func:
        # workaround to enable empty decorator
        return functools.partial(strategy, names=names)

    return wrapper_delegate(func, names)


def context(func):
    """
    Context decorator to inject the default context instance.
    :param func: original function instance
    :return:
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        config = Config.get_instance()
        registry = Registry.get_instance()
        instance = args[0]

        setattr(instance, 'config', config)
        setattr(instance, 'registry', registry)

        return func(*args, **kwargs)
    return wrapper


def config(func=None, path: str = None, group: str = None):
    """
    Config decorator to import and inject the specified config.
    This config instance can be grouped in a defined group name.
    The default config containing this new property is injected into an object instance.
    If none of the options is set, then only the default config if injected.
    :param func: original function instance
    :param path: path of the config file
    :param group: group name to categorize new config instance
    :return:
    """
    if not func:
        # workaround to enable empty decorator
        return functools.partial(config, path=path, group=group)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        config = Config.get_instance()
        instance = args[0]

        if path is not None:
            imported_config = Config.load(path=path)
            if group is not None:
                config.set_value(group, imported_config)
            else:
                for k, v in imported_config.items():
                    setattr(config, k, imported_config)

        setattr(instance, 'config', config)

        return func(*args, **kwargs)
    return wrapper


def search_and_load_type(instance):
    registry = Registry.get_instance()
    reg_instance, key = DotDict.resolve(registry.types, instance)
    class_ = getattr(reg_instance, key, None)
    if class_ is None or not inspect.isclass(class_):
        raise DependencyInjectionError()
    new_ = class_()
    registry.register_instance(key, new_)
    return new_


def inject(instance: str, name: str, type: RegistryOption = RegistryOption.Instances):
    """
    Injects a named variable of any given instance into the decorated object instance.
    :param instance: Source instance available in the registry.
    :param name: Target name of the variable to inject.
    :param type: Option for the registry lookup.
    :return:
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            registry = Registry.get_instance()
            obj_instance = args[0]

            value = None
            # if instances option selected lookup attribute or load if not already instantiated
            if type == RegistryOption.Instances:
                reg_instance, key = DotDict.resolve(registry.instances, instance)
                value = getattr(reg_instance, key, None)
                if value is None:
                    value = search_and_load_type(instance)
            # otherwise if types then return simple type
            elif type == RegistryOption.Types:
                reg_instance, key = DotDict.resolve(registry.types, instance)
                value = getattr(reg_instance, key, None)

            # if value is still None then through error, since it was never registered or failed to load
            if value is None:
                raise DependencyInjectionError()

            setattr(obj_instance, name, value)

            return func(*args, **kwargs)
        return wrapper
    return decorator


def device(func=None, id: str = None, group: str = 'device.default', name: str = 'device'):
    """
    Injects a device info into the current object instance.
    :param func: original function instance
    :param id: device id if pre-specified
    :param group: group path for the config
    :param name: device name for injection
    :return:
    """
    if not func:
        # workaround to enable empty decorator
        return functools.partial(device, id=id, group=group, name=name)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        config = Config.get_instance()
        obj_instance = args[0]

        # update device according to id
        if id is not None:
            if not torch.cuda.is_available() or 'cpu' in id:
                config.set_value(group, torch.device('cpu'))
            else:
                config.set_value(group, torch.device(id))

        value = config.get_value(group)
        # if never set but called, then through exception
        if value is None:
            raise DependencyInjectionError()

        setattr(obj_instance, name, value)

        return func(*args, **kwargs)
    return wrapper
