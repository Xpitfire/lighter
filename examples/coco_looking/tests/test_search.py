from lighter.context import Context
from lighter.parameter import GridParameter, BinaryParameter, StrategyParameter, Parameter

from lighter.decorator import search


class SearchExperiment:
    @search(group='sgd',
            params=[('lr', GridParameter(ref='optimizer.lr', min=0.0005, max=0.5, step=0.0005)),
                    ('weight_decay', GridParameter(ref='optimizer.lr', min=0.0, max=1.0, step=1.0)),
                    ('pretrained', BinaryParameter(ref='model.pretrained')),
                    ('hidden_units', GridParameter(ref='model.hidden_units', min=100, max=1000, step=200)),
                    ('strategy', StrategyParameter(ref='configs/sgd',
                                                   options=['alexnet.modules.config.json',
                                                            'inception.modules.config.json',
                                                            'resnet.modules.config.json']))])
    @search(group='adam',
            params=[('lr', GridParameter(ref='optimizer.lr', min=0.0003, max=0.005, step=0.0003)),
                    ('pretrained', BinaryParameter(ref='model.pretrained')),
                    ('hidden_units', GridParameter(ref='model.hidden_units', min=100, max=1000, step=200)),
                    ('strategy', StrategyParameter(ref='configs/adam',
                                                   options=['alexnet.modules.config.json',
                                                            'inception.modules.config.json',
                                                            'resnet.modules.config.json']))])
    def __init__(self):
        pass

    def run(self):
        for group in self.search.keys():
            obj = self.search[group]
            if isinstance(obj, dict):
                for key in obj.keys():
                    if issubclass(type(obj[key]), Parameter):
                        param = obj[key]
                        print(param)


if __name__ == '__main__':
    Context.create()
    se = SearchExperiment()
    se.run()
