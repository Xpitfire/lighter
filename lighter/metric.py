from lighter.decorator import context


class BaseMetric(object):
    @context
    def __init__(self):
        pass
