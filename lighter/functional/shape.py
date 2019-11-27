import torch


def one_hot(y, num_classes):
    y_onehot = torch.FloatTensor(y.shape[0], num_classes)
    y_onehot.zero_()
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    return y_onehot
