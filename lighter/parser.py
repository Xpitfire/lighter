import os
import shutil

from lighter.config import Config
from lighter.search import ParameterSearch
from lighter.decorator import context
from lighter.parameter import Parameter


class ParameterParser:
    @context
    def __init__(self, output_path: str = 'runs/search'):
        self.output_path = os.path.join(output_path, self.config.context_id)
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path, ignore_errors=False, onerror=None)
        os.makedirs(self.output_path)

    def run(self):
        for group in self.search.keys():
            params = []
            obj = self.search[group]
            if isinstance(obj, dict):
                for key in obj.keys():
                    if issubclass(type(obj[key]), Parameter):
                        param = obj[key]
                        params.append(param)
            configs = ParameterSearch.compile(params, Config.get_instance())
            for config in configs:
                file_name = os.path.join(self.output_path, '{}.json'.format(config.experiment_id))
                config.save(file_name)
