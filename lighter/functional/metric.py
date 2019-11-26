def TP(target, prediction):
    """
    True positives.
    :param target: target value
    :param prediction: prediction value
    :return:
    """
    return (target.float() * prediction.float().round()).sum()


def TN(target, prediction):
    """
    True negatives.
    :param target: target value
    :param prediction: prediction value
    :return:
    """
    return ((target == 0).float() * (prediction.float().round() == 0).float()).sum()


def FP(target, prediction):
    """
    False positives.
    :param target: target value
    :param prediction: prediction value
    :return:
    """
    return ((target == 0).float() * prediction.float().round()).sum()


def FN(target, prediction):
    """
    False negatives.
    :param target: target value
    :param prediction: prediction value
    :return:
    """
    return (target.float() * (prediction.float().round() == 0).float()).sum()


def TPR(target, prediction, eps=1e-7):
    """
    True positive rate metric.
    :param target: target value
    :param prediction: prediction value
    :param eps: epsilon to avoid zero division
    :return:
    """
    tp = TP(target, prediction)
    fn = FN(target, prediction)
    s = tp + fn + eps
    return tp / s


def TNR(target, prediction, eps=1e-7):
    """
    True negative rate metric.
    :param target: target value
    :param prediction: prediction value
    :param eps: epsilon to avoid zero division
    :return:
    """
    tn = TN(target, prediction)
    fp = FP(target, prediction)
    s = (tn + fp + eps)
    assert (s > 0)
    return tn / s


def FPR(target, prediction, eps=1e-7):
    """
    False positive rate metric.
    :param target: target value
    :param prediction: prediction value
    :param eps: epsilon to avoid zero division
    :return:
    """
    fp = FP(target, prediction)
    tn = TN(target, prediction)
    s = fp + tn + eps
    assert (s > 0)
    return fp / s


def FNR(target, prediction, eps=1e-7):
    """
    False negative rate metric.
    :param target: target value
    :param prediction: prediction value
    :param eps: epsilon to avoid zero division
    :return:
    """
    fn = FN(target, prediction)
    tp = TP(target, prediction)
    s = fn + tp + eps
    assert (s > 0)
    return fn / s


def ACC(target, prediction, eps=1e-7):
    """
    Accuracy metric.
    :param target: target value
    :param prediction: prediction value
    :param eps: epsilon to avoid zero division
    :return:
    """
    tp = TP(target, prediction)
    tn = TN(target, prediction)
    p = target.sum().float()
    n = (target == 0).sum().float()
    s = p + n + eps
    assert (s > 0)
    return (tp + tn) / s


def BACC(target, prediction, eps=1e-7):
    """
    Balanced accuracy metric.
    :param target: target value
    :param prediction: prediction value
    :param eps: epsilon to avoid zero division
    :return:
    """
    tp = TP(target, prediction)
    tn = TN(target, prediction)
    p = target.sum().float()
    n = (target == 0).sum().float()
    s = (p + eps) + tn / (n + eps)
    assert (s > 0)
    return 0.5 * (tp / s)


def precision(target, prediction, eps=1e-7):
    """
    Precision metric.
    :param target: target value
    :param prediction: prediction value
    :param eps: epsilon to avoid zero division
    :return:
    """
    tp = TP(target, prediction)
    fp = FP(target, prediction)
    s = (tp + fp + eps)
    assert (s > 0)
    return tp / s


def recall(target, prediction, eps=1e-7):
    """
    Recall metric.
    :param target: target value
    :param prediction: prediction value
    :param eps: epsilon to avoid zero division
    :return:
    """
    tp = TP(target, prediction)
    fn = FN(target, prediction)
    s = (tp + fn + eps)
    assert (s > 0)
    return tp / s


def f1_score(target, prediction, eps=1e-7):
    """
    F1-score metric.
    :param target: target value
    :param prediction: prediction value
    :param eps: epsilon to avoid zero division
    :return:
    """
    precision_ = precision(target, prediction)
    recall_ = recall(target, prediction)
    n = (precision_ * recall_)
    s = (precision_ + recall_ + eps)
    assert (s > 0)
    return 2 * (n / s)
