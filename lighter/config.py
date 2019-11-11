import json
import os
import argparse
from threading import RLock
import logging

import torch

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
            value = Loader.get_instance().import_path(value[len("type::"):])
            res = Registry.get_instance().contains_type(name)
            # check if types where already registered
            if not res:
                # register type to registry
                Registry.get_instance().register_type(name, value)

        except ModuleNotFoundError as e:
            logging.warning("Error while importing '{}' - {}".format(value, e))
    # import config if config:: reference was found in json
    elif isinstance(value, str) and "config::" in value:
        try:
            conf = Config(value[len("config::"):])
            conf = DotDict(conf)
            conf['config_path'] = value[len("config::"):]
            value = conf
        except ModuleNotFoundError as e:
            logging.warning("Error while importing '{}' - {}".format(value, e))
    # else regular set
    elif isinstance(value, dict):
        for k, v in value.items():
            value[k] = import_value_rec(k, v)
        value = DotDict(value)

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
            conf = Config(value[len("import::"):])
            for k, v in DotDict(conf).items():
                setattr(parent, k, v)
                logging.info("Config: {}={}".format(k, getattr(parent, k)))
        # decent down the dictionary instance
        elif isinstance(value, dict):
            dict_ = DotDict(value)
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


class Config(DotDict):
    _instance = None
    # thread save lock to ensure single instance creation for the default config
    _mutex = RLock()

    def __init__(self, filename: str = None, override_args=None, **kwargs):
        """
        Create config object from json file.

        ATTENTION: Only register types but never instantiate class objects in the config instance, since they might
        require the config context before it has every been built!

        :param filename: optional;
            If passed read config from specified file, otherwise parse command line for config parameter and optionally
            override arguments.
        :param override_args: args to override the current properties
        """
        super(Config, self).__init__(**kwargs)
        # Read config and override with args if passed
        if filename is not None and os.path.exists(filename):
            with open(filename) as f:
                self.initialize_from_json(json.loads(f.read()).items())
            # override if necessary
            if override_args is not None:
                self.override_from_commandline(override_args)

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
    def create_instance(config_file: str = None):
        """
        Create default config instance. Loads the command line overridable config settings and
        sets the default device instance.
        :param config_file: path to config file
        :return:
        """
        Config._mutex.acquire()
        try:
            if Config._instance is None:
                Config._instance, args = Config.load(config_file, parse_args_fallback=True)
                # load default device
                device = args.device
                if torch.cuda.is_available() and 'cuda' in device:
                    Config._instance.set_value('device.default', torch.device(device))
                else:
                    Config._instance.set_value('device.default', torch.device('cpu'))
            return Config._instance
        finally:
            Config._mutex.release()

    @staticmethod
    def get_instance():
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
    def load(path: str = None, parse_args_fallback: bool = False):
        """
        Loads a property file and allows to override arguments based on command line overrides.
        :param path: path to property file
        :param parse_args_fallback: fallback to --config if no path was specified.
        :return: Config instance or Config with command line args tuple if parse_args_fallback is enabled
        """
        if parse_args_fallback:
            parser = argparse.ArgumentParser()
            parser.add_argument('--config', type=str, help='path to config file')
            parser.add_argument('--device', type=str, default='cuda', help='set the used device')
            args = parser.parse_args()
            if args.config is not None:
                path = args.config
            return Config(path), args
        return Config(path)
