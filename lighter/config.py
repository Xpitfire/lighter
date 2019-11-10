import json
import os
import argparse

from threading import RLock

import logging

from lighter.loader import Loader
from lighter.misc import extract_named_args, try_to_number_or_bool, DotDict
from lighter.registry import Registry


def import_value_rec(name, value):
    if isinstance(value, str) and "class::" in value:
        try:
            value = Loader.get_instance().import_path(value[len("class::"):])
            Registry.get_instance().register_type(name, value)
        except ModuleNotFoundError:
            logging.warning("Error while importing '{}'".format(value))
    elif isinstance(value, str) and "config::" in value:
        try:
            conf = Config(value[len("config::"):])
            conf = DotDict(conf)
            conf['config_path'] = value[len("config::"):]
            value = conf
        except ModuleNotFoundError:
            logging.warning("Error while importing '{}'".format(value))
    elif isinstance(value, dict):
        for k, v in value.items():
            value[k] = import_value_rec(k, v)
        value = DotDict(value)

    return value


def override(parent, name, value):
    if value is not None:
        if isinstance(value, str) and "import::" in value:
            conf = Config(value[len("import::"):])
            for k, v in DotDict(conf).items():
                setattr(parent, k, v)
                logging.info("Config: {}={}".format(k, getattr(parent, k)))
        elif isinstance(value, dict):
            dict_ = DotDict(value)
            for k, v in dict_.copy().items():
                override(dict_, k, v)
                if isinstance(v, str) and "import::" in v:
                    del dict_[k]
            setattr(parent, name, dict_)
        else:
            value = import_value_rec(name, value)
            setattr(parent, name, value)
            logging.info("Config: {}={}".format(name, getattr(parent, name)))


class Config(DotDict):
    _instance = None
    _mutex = RLock()

    def __init__(self, filename: str = None, override_args=None, **kwargs):
        """Create config object from json file.

        filename : optional;
            If passed read config from specified file, otherwise parse command line for config parameter and optionally
            override arguments.
        """
        super(Config, self).__init__(**kwargs)
        # Read config and override with args if passed
        if filename is not None and os.path.exists(filename):
            with open(filename) as f:
                self.initialize_from_json(json.loads(f.read()).items())
            # override if necessary
            if override_args is not None:
                self.override_from_commandline(override_args)

    @staticmethod
    def resolve(parent, name):
        groups = name.split('.')
        for group in groups[:-1]:
            parent = parent.get_value(group)
        return parent, groups[-1]

    def set_value(self, name, value):
        parent, name = Config.resolve(self, name)
        override(parent, name, value)

    def has_value(self, name):
        parent, name = Config.resolve(self, name)
        return hasattr(parent, name)

    def get_value(self, name, default=None):
        parent, name = Config.resolve(self, name)
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
        Config._mutex.acquire()
        try:
            if Config._instance is None:
                Config._instance = Config.load(config_file, parse_args_fallback=True)
            return Config._instance
        finally:
            Config._mutex.release()

    @staticmethod
    def get_instance():
        return Config._instance

    @staticmethod
    def load(path: str = None, parse_args_fallback: bool = False):
        if path is None and parse_args_fallback:
            parser = argparse.ArgumentParser()
            parser.add_argument('--config', type=str, help='path to config file')
            args = parser.parse_args()
            path = args.config
        return Config(path)
