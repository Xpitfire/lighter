from torchvision.transforms import Compose


class BaseTransform(Compose):
    """Base class for transforming data.
    """
    def __init__(self, transforms):
        super(BaseTransform, self).__init__(transforms)
