import functools

from lighter.config import Config
from lighter.exceptions import DependencyInjectionError
from lighter.registry import Registry


def experiment(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        config = Config.get_instance()
        registry = Registry.get_instance()
        instance = args[0]

        setattr(instance, 'config', config)
        setattr(instance, 'registry', registry)

        for name in ['model',
                     'dataset',
                     'data_builder',
                     'optimizer',
                     'collectible',
                     'criterion',
                     'metric',
                     'writer']:
            value = getattr(registry.instances, name, None)
            if value is None:
                raise DependencyInjectionError()
            setattr(instance, name, value)

        result = func(*args, **kwargs)
        return result
    return wrapper


def wrapper_delegate(func, name):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        registry = Registry.get_instance()
        instance = args[0]

        value = getattr(registry.instances, name, None)
        if value is None:
            raise DependencyInjectionError()

        setattr(instance, name, value)

        result = func(*args, **kwargs)
        return result
    return wrapper


def dataset(func, name: str = None):
    if name is None:
        name = 'dataset'
    return wrapper_delegate(func, name)


def model(func, name: str = None):
    if name is None:
        name = 'model'
    return wrapper_delegate(func, name)


def context(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        config = Config.get_instance()
        registry = Registry.get_instance()
        instance = args[0]

        setattr(instance, 'config', config)
        setattr(instance, 'registry', registry)

        result = func(*args, **kwargs)
        return result
    return wrapper


def config(path: str = None, group: str = None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            instance = args[0]

            config = Config.get_instance()
            if path is not None:
                imported_config = Config.load(path=path)
                if group is not None:
                    config.set_value(group, imported_config)
                else:
                    for k, v in imported_config.items():
                        setattr(config, k, imported_config)

            setattr(instance, 'config', config)

            result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator
