import json
import os
import argparse
from threading import RLock
import logging
import torch
from box import Box
from lighter.exceptions import InvalidTypeReferenceError
from lighter.loader import Loader
from lighter.misc import extract_named_args, try_to_number_or_bool, DotDict
from lighter.registry import Registry


def import_value_rec(name, value):
    """
    Imports configs or classes to the current config instance.
    :param name: property name
    :param value: property value
    :return: value instance of the replaced property
    """
    # import classes if type:: reference was found in json
    if isinstance(value, str) and "type::" in value:
        try:
            type_ = Loader.import_path(value[len("type::"):])
            if type_ is None:
                raise InvalidTypeReferenceError('Config: Could not find specified reference: {}'.format(value))

            # register type to registry
            Registry.get_instance().register_type(name, type_)

        except ModuleNotFoundError as e:
            logging.warning("Error while importing '{}' - {}".format(value, e))
    # import config if config:: reference was found in json
    elif isinstance(value, str) and "config::" in value:
        try:
            conf = Config(path=value[len("config::"):])
            conf = Box(conf)
            conf['config_path'] = value[len("config::"):]
            value = conf
        except ModuleNotFoundError as e:
            logging.warning("Error while importing '{}' - {}".format(value, e))
    # else regular set
    elif isinstance(value, dict):
        for k, v in value.items():
            value[k] = import_value_rec(k, v)
        value = Box(value)

    return value


def override(parent, name, value):
    """
    Overrides a config key.
    :param parent: Parent dictionary instance
    :param name: name of the property
    :param value: value for the property
    :return:
    """
    if value is not None:
        # import key-values into the current property instance
        if isinstance(value, str) and "import::" in value:
            conf = Config(path=value[len("import::"):])
            for k, v in Box(conf).items():
                setattr(parent, k, v)
                logging.info("Config: {}={}".format(k, getattr(parent, k)))
        # decent down the dictionary instance
        elif isinstance(value, dict):
            dict_ = Box(value)
            for k, v in dict_.copy().items():
                override(dict_, k, v)
                # remove old import:: reference after import
                if isinstance(v, str) and "import::" in v:
                    del dict_[k]
            setattr(parent, name, dict_)
        else:
            value = import_value_rec(name, value)
            setattr(parent, name, value)
            logging.info("Config: {}={}".format(name, getattr(parent, name)))


class Config(Box):
    _instance = None
    # thread save lock to ensure single instance creation for the default config
    _mutex = RLock()

    def __init__(self, path: str = None, override_args=None, **kwargs):
        """
        Create config object from json file.

        :param path: optional;
            If passed read config from specified file, otherwise parse command line for config parameter and optionally
            override arguments.
        :param override_args: args to override the current properties
        """
        super(Config, self).__init__()
        if kwargs:
            self.initialize_from_json(kwargs.items())
        # Read config and override with args if passed
        if path is not None:
            if not os.path.exists(path):
                raise FileNotFoundError('Invalid config path! Could not resolve: {}'.format(path))
            with open(path) as f:
                self.initialize_from_json(json.loads(f.read()).items())
            # override if necessary
            if override_args is not None:
                self.override_from_commandline(override_args)

    def copy(self) -> "Config":
        config = Config()
        dict_ = super().copy()
        config.initialize_from_dict(dict_)
        return config

    def save(self, file_name):
        dict_str = json.dumps(self, indent=2)
        with open(file_name, 'w') as file:
            file.write(dict_str)

    def set_value(self, name, value):
        """Sets the properties recursively according to a dot-separated reference.
        """
        parent, name = DotDict.resolve(self, name)
        override(parent, name, value)

    def has_value(self, name):
        """Checks properties recursively according to a dot-separated reference.
        """
        parent, name = DotDict.resolve(self, name)
        return hasattr(parent, name)

    def get_value(self, name, default=None):
        """Returns the properties recursively according to a dot-separated reference.
        """
        parent, name = DotDict.resolve(self, name)
        return getattr(parent, name, default)

    def initialize_from_dict(self, kv_pairs):
        if kv_pairs:
            for i, (name, value) in enumerate(kv_pairs.items()):
                override(self, name, value)

    def initialize_from_json(self, nv_pairs=None):
        if nv_pairs:
            for i, (name, value) in enumerate(nv_pairs):
                override(self, name, value)

    def override_from_commandline(self, override_args=None):
        if override_args is not None:
            override_dict = extract_named_args(override_args)
            for k, v in override_dict.items():
                name = k[2:] if "--" in k else k  # remove leading --
                if v is None:
                    value = True  # assume cmd switch
                else:
                    value = v if v.startswith('"') or v.startswith("'") else try_to_number_or_bool(v)
                if "." in name:
                    names = name.split(".")
                    name = names[0]
                    if len(names) == 2:
                        if hasattr(self, names[0]):
                            curdict = getattr(self, names[0])
                        else:
                            curdict = dict()
                        curdict[names[1]] = value
                        value = curdict
                    elif len(names) == 3:
                        if hasattr(self, names[0]):
                            curdict = getattr(self, names[0])
                        else:
                            curdict = dict()

                        if names[1] in curdict:
                            subdict = curdict[names[1]]
                        else:
                            curdict[names[1]] = dict()
                            subdict = curdict[names[1]]

                        subdict[names[2]] = value
                        value = curdict
                    else:
                        raise Exception("Unsupported command line option (can only override dicts with 1 or 2 levels)")
                override(self, name, value)

    @staticmethod
    def create_instance(config_file: str = None,
                        config_dict: dict = None,
                        parse_args_override: bool = True,
                        device: str = None) -> "Config":
        """
        Create default config instance. Loads the command line overridable config settings and
        sets the default device instance.
        :param config_file: path to config file
        :param config_dict: pre-configs
        :param parse_args_override: override
        :param device: training device
        :return:
        """
        if config_dict is None:
            config_dict = {}
        Config._mutex.acquire()
        try:
            Config._instance, args = Config.load(path=config_file,
                                                 parse_args_override=parse_args_override,
                                                 **config_dict)
            # load default device
            if args is not None and args.device is not None:
                Config._instance.set_value('device.default', args.device)
            elif device is not None:
                Config._instance.set_value('device.default', device)
            elif torch.cuda.is_available():
                Config._instance.set_value('device.default', 'cuda')
            else:
                Config._instance.set_value('device.default', 'cpu')
            return Config._instance
        finally:
            Config._mutex.release()

    @staticmethod
    def get_instance() -> "Config":
        """
        Return default config instance.
        :return:
        """
        Config._mutex.acquire()
        try:
            return Config._instance
        finally:
            Config._mutex.release()

    @staticmethod
    def load(path: str = None, parse_args_override: bool = False, **kwargs):
        """
        Loads a property file and allows to override arguments based on command line overrides.
        :param path: path to property file
        :param parse_args_override: fallback to --config if no path was specified.
        :return: Config instance or Config with command line args tuple if parse_args_fallback is enabled
        """
        if parse_args_override:
            parser = argparse.ArgumentParser()
            parser.add_argument('--config', type=str, help='path to config file')
            parser.add_argument('--device', type=str, help='set the used device')
            args = parser.parse_args()
            if args.config is not None:
                path = args.config
            return Config(path=path, **kwargs), args
        return Config(path=path, **kwargs), None
