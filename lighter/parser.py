import os
import shutil
from lighter.search import ParameterSearch
from lighter.decorator import context
from lighter.parameter import Parameter


class ConfigParser:
    """
    The config parser creates a set of parameters based on the decorated experiment
    class. It permutes all possible combinations within a group alias and saves the values to the
    directory.
    """
    @context
    def __init__(self, experiment=None, output_path: str = 'runs/search'):
        self.output_path = os.path.join(output_path, self.config.context_id)
        self.experiment = experiment
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path, ignore_errors=False, onerror=None)
        os.makedirs(self.output_path)

    def parse(self, persist: bool = True):
        """
        Parses a list of configs which can be used for training different strategies.
        :param persist: bool value which allows to persist the configs to the file system
        :return: list of parameters
        """
        list_of_configs = []
        for group in self.search.keys():
            params = []
            obj = self.search[group]
            if isinstance(obj, dict):
                for key in obj.keys():
                    if issubclass(type(obj[key]), Parameter):
                        param = obj[key]
                        params.append(param)
            configs = ParameterSearch.compile(params, self.experiment.config)
            list_of_configs.extend(configs)
            if persist:
                for config in configs:
                    file_name = os.path.join(self.output_path, '{}.json'.format(config.experiment_id))
                    config.save(file_name)
        return list_of_configs
