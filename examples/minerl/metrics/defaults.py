from lighter.metric import BaseMetric


def P(target):
    return target.sum()


def N(target):
    n = (target == 0).sum()
    return n.float()


def TP(target, prediction):
    tp = (target * prediction.round()).sum()
    return tp


def TN(target, prediction):
    tn = ((target == 0).float() * (prediction.round() == 0).float()).sum()
    return tn


def FP(target, prediction):
    fp = ((target == 0).float() * prediction.round()).sum()
    return fp


def FN(target, prediction):
    fn = (target * (prediction.round() == 0).float()).sum()
    return fn


def TPR(target, prediction, eps=1e-7):
    tp = TP(target, prediction)
    fn = FN(target, prediction)
    s = tp + fn + eps
    tpr = tp / s
    return tpr


def TNR(target, prediction, eps=1e-7):
    tn = TN(target, prediction)
    fp = FP(target, prediction)
    s = tn + fp + eps
    tnr = tn / s
    return tnr


def FPR(target, prediction, eps=1e-7):
    fp = FP(target, prediction)
    tn = TN(target, prediction)
    s = fp + tn + eps
    assert (s > 0)
    fpr = fp / s
    return fpr


def FNR(target, prediction, eps=1e-7):
    fn = FN(target, prediction)
    tp = TP(target, prediction)
    s = fn + tp + eps
    fnr = fn / s
    return fnr


def ACC(target, prediction, eps=1e-7):
    tp = TP(target, prediction)
    tn = TN(target, prediction)
    p = P(target)
    n = N(target)
    s = p + n + eps
    acc = (tp + tn) / s
    return acc


def BACC(target, prediction, eps=1e-7):
    tp = TP(target, prediction)
    tn = TN(target, prediction)
    p = P(target)
    n = N(target)
    bacc = tp / (p + eps) + tn / (n + eps)
    bacc = 0.5 * bacc
    return bacc


class Metric(BaseMetric):
    def __call__(self, pred, target):
        return {
            'acc': ACC(target, pred),
            'bacc': BACC(target, pred),
            'tpr': TPR(target, pred),
            'tnr': TNR(target, pred),
            'fpr': FPR(target, pred),
            'fnr': FNR(target, pred)
        }
