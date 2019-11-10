from lighter.decorator import dataset, context


class BaseDataBuilder(object):
    @dataset
    @context
    def __init__(self):
        pass

    def loader(self, **kwargs):
        raise NotImplementedError('Cannot instantiate builder base class.')
