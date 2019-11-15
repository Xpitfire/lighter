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
            'acc': ACC(target, prediction),
            'bacc': BACC(target, prediction),
            'precision': precision(target, prediction),
            'recall': recall(target, prediction),
            'f1_score': f1_score(target, prediction),
            'tpr': TPR(target, prediction),
            'tnr': TNR(target, prediction),
            'fpr': FPR(target, prediction),
            'fnr': FNR(target, prediction)
        }
