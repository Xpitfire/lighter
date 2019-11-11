import functools
from lighter.config import Config
from lighter.exceptions import DependencyInjectionError
from lighter.registry import Registry


def experiment(func, injectables: list = None):
    """
    Experiment decorator to inject multiple values at once.
    :param func: original decorated function
    :param injectables: list of values to inject
    :return:
    """
    if injectables is None:
        injectables = ['model', 'dataset', 'data_builder', 'optimizer',
                       'collectible', 'criterion', 'metric', 'writer']

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

        result = func(*args, **kwargs)
        return result
    return wrapper


def wrapper_delegate(func, name):
    """
    Delegate wrapper to inject values.
    :param func: original decorated function
    :param name: name of the value to inject
    :return:
    """
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
    """
    Dataset decorator to inject the default dataset instance.
    :param func: original decorated function
    :param name: name of the injection variable as specified in the config
    :return:
    """
    if name is None:
        name = 'dataset'
    return wrapper_delegate(func, name)


def model(func, name: str = None):
    """
    Model decorator to inject the default model instance.
    :param func: original decorated function
    :param name: name of the injection variable as specified in the config
    :return:
    """
    if name is None:
        name = 'model'
    return wrapper_delegate(func, name)


def context(func):
    """
    Context decorator to inject the default context instance.
    :param func: original decorated function
    :return:
    """
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
    """
    Config decorator to import and inject the specified config.
    This config instance can be grouped in a defined group name.
    The default config containing this new property is injected into an object instance.
    If none of the options is set, then only the default config if injected.
    :param path: path of the config file
    :param group: group name to categorize new config instance
    :return:
    """
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
