import os
from lighter.search import ParameterSearch
from lighter.decorator import context
from lighter.parameter import Parameter


class ParameterParser:
    @context
    def __init__(self, output_path: str = 'runs/search'):
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    def run(self):
        for group in self.search.keys():
            params = []
            obj = self.search[group]
            if isinstance(obj, dict):
                for key in obj.keys():
                    if issubclass(type(obj[key]), Parameter):
                        param = obj[key]
                        params.append(param)
            configs = ParameterSearch.compile(params)
            for config in configs:
                file_name = os.path.join(self.output_path, '{}.json'.format(config.experiment_id))
                config.save(file_name)
