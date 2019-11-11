from lighter.decorator import dataset, context


class BaseDataBuilder(object):
    """
    Base class for building the data loaders with injected 'dataset' reference and application 'context'.
    """
    @dataset
    @context
    def __init__(self):
        pass

    def loader(self, *args, **kwargs):
        """
        Returns the data loader instances.
        :param args:
        :param kwargs:
        :return: data loader(s) for an experiment.
        """
        raise NotImplementedError('Cannot instantiate builder base class.')
