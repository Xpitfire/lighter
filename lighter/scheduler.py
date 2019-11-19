import os

from lighter.search import ParameterSearch

from lighter.context import Context
from lighter.loader import Loader
from lighter.config import Config
from lighter.registry import Registry


class Scheduler(object):
    def __init__(self, path: str, experiment: str):
        self.files = [os.path.join(path, file) for file in os.listdir(path)]
        self.experiment = Loader.import_path(experiment)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < len(self.files):
            context = Context.create(self.files[self.idx],
                                     parse_args_override=True,
                                     instantiate_types=False)
            experiment = self.experiment()

            context.registry = Registry.create_instance()
            config = Config.create_instance(self.files[self.idx],
                                            parse_args_override=True)
            context.config = config
            context.instantiate_types(context.registry.types)

            setattr(experiment, 'config', config)
            search = ParameterSearch.create_instance()
            setattr(experiment, 'search', search)
            for key in config['strategy'].keys():
                setattr(experiment, key, context.registry.instances[key])

            self.idx += 1
            return experiment
        else:
            raise StopIteration
