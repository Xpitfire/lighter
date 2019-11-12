from lighter.decorator import context


class BaseTransform(object):
    @context
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError('BaseTransform: No implementation found!')
