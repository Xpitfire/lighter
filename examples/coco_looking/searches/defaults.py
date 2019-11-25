from lighter.decorator import search, config
from lighter.parameter import GridParameter, BinaryParameter, StrategyParameter, ListParameter, SetParameter


class ParameterSearchRegistration:
    @search(group='sgd',
            params=[('lr', GridParameter(ref='optimizer.lr', min=0.001, max=0.005, step=0.001)),
                    ('weight_decay', ListParameter(ref='optimizer.weight_decay', options=[0.0, 0.9])),
                    ('freeze_pretrained', BinaryParameter(ref='model.freeze_pretrained')),
                    ('hidden_units', GridParameter(ref='model.hidden_units', min=100, max=200, step=100)),
                    ('optimizer', SetParameter(ref='strategy.optimizer', option="type::optimizers.defaults.Optimizer")),
                    ('model_output', SetParameter(ref='model.output', option=1)),
                    ('model_pretrained', SetParameter(ref='model.pretrained', option=True)),
                    ('strategy', StrategyParameter(ref='strategy', options=['searches/coco_looking.config.json'])),
                    ('model', ListParameter(ref='strategy.model',
                                            options=['type::models.alexnet.AlexNetFeatureExtractionModel',
                                                     'type::models.resnet.ResNetFeatureExtractionModel']))])
    @search(group='adam',
            params=[('lr', ListParameter(ref='optimizer.lr', options=[1e-3, 1e-5, 0.0025])),
                    ('freeze_pretrained', BinaryParameter(ref='model.freeze_pretrained')),
                    ('hidden_units', GridParameter(ref='model.hidden_units', min=100, max=400, step=100)),
                    ('model_output', SetParameter(ref='model.output', option=1)),
                    ('model_pretrained', SetParameter(ref='model.pretrained', option=True)),
                    ('optimizer', SetParameter(ref='strategy.optimizer', option="type::optimizers.adam.Optimizer")),
                    ('strategy', StrategyParameter(ref='strategy', options=['searches/coco_looking.config.json'])),
                    ('model', ListParameter(ref='strategy.model',
                                            options=['type::models.alexnet.AlexNetFeatureExtractionModel',
                                                     'type::models.resnet.ResNetFeatureExtractionModel']))])
    @config(path='data_builders/defaults.config.json', property='data_builder')
    @config(path='datasets/defaults.config.json', property='dataset')
    @config(path='optimizers/defaults.config.json', property='optimizer')
    def __init__(self):
        pass
