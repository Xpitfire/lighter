from lighter.decorator import context
from lighter.functional.metric import *


class BaseMetric(object):
    """
    Base metric class to register the required losses.
    """
    @context
    def __init__(self):
        pass

    def __call__(self, prediction, target):
        return {
            'acc': ACC(target, prediction).item(),
            'bacc': BACC(target, prediction).item(),
            'precision': precision(target, prediction).item(),
            'recall': recall(target, prediction).item(),
            'f1_score': f1_score(target, prediction).item(),
            'tpr': TPR(target, prediction).item(),
            'tnr': TNR(target, prediction).item(),
            'fpr': FPR(target, prediction).item(),
            'fnr': FNR(target, prediction).item()
        }
