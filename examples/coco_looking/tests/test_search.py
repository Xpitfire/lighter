from lighter.parser import ParameterParser
from lighter.context import Context
from lighter.parameter import GridParameter, BinaryParameter, StrategyParameter
from lighter.decorator import search


class SearchExperiment:
    @search(group='sgd',
            params=[('lr', GridParameter(ref='optimizer.lr', min=0.0001, max=0.002, step=0.001)),
                    ('weight_decay', GridParameter(ref='optimizer.weight_decay', min=0.0, max=1.0, step=1.0)),
                    ('pretrained', BinaryParameter(ref='model.freeze_pretrained')),
                    ('hidden_units', GridParameter(ref='model.hidden_units', min=100, max=200, step=100)),
                    ('strategy', StrategyParameter(ref='configs/sgd',
                                                   options=['alexnet.modules.config.json',
                                                            'inception.modules.config.json',
                                                            'resnet.modules.config.json']))])
    @search(group='adam',
            params=[('lr', GridParameter(ref='optimizer.lr', min=0.01, max=0.03, step=0.01)),
                    ('pretrained', BinaryParameter(ref='model.freeze_pretrained')),
                    ('hidden_units', GridParameter(ref='model.hidden_units', min=100, max=200, step=100)),
                    ('strategy', StrategyParameter(ref='configs/adam',
                                                   options=['alexnet.modules.config.json',
                                                            'inception.modules.config.json',
                                                            'resnet.modules.config.json']))])
    def __init__(self):
        pass


if __name__ == '__main__':
    Context.create()
    SearchExperiment()
    sp = ParameterParser()
    sp.run()
