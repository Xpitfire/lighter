from lighter.decorator import context


class BaseMetric(object):
    """
    Base metric class to register the required losses.
    """
    @context
    def __init__(self):
        pass
