from lighter.experiment import DefaultExperiment
from lighter.parser import ParameterParser
from lighter.context import Context
from lighter.parameter import GridParameter, BinaryParameter, StrategyParameter, ListParameter

from lighter.decorator import search, strategy


class SearchExperiment(DefaultExperiment):
    @search(group='sgd',
            params=[('lr', GridParameter(ref='optimizer.lr', min=0.001, max=0.005, step=0.001)),
                    ('weight_decay', ListParameter(ref='optimizer.weight_decay', options=[0.0, 0.9])),
                    ('pretrained', BinaryParameter(ref='model.freeze_pretrained')),
                    ('hidden_units', GridParameter(ref='model.hidden_units', min=100, max=200, step=100)),
                    ('strategy', StrategyParameter(ref='strategy',
                                                   options=['configs/sgd/alexnet.modules.config.json',
                                                            'configs/sgd/inception.modules.config.json',
                                                            'configs/sgd/resnet.modules.config.json']))])
    @search(group='adam',
            params=[('lr', ListParameter(ref='optimizer.lr', options=[1e-3, 1e-5, 0.0025])),
                    ('hidden_units', GridParameter(ref='model.hidden_units', min=100, max=400, step=100)),
                    ('strategy', StrategyParameter(ref='strategy',
                                                   options=['configs/adam/alexnet.modules.config.json',
                                                            'configs/adam/inception.modules.config.json',
                                                            'configs/adam/resnet.modules.config.json']))])
    @strategy(config='configs/coco_looking.config.json')
    def __init__(self):
        super(SearchExperiment, self).__init__()


if __name__ == '__main__':
    Context.create()
    SearchExperiment()
    sp = ParameterParser()
    sp.run()
