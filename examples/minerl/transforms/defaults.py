from torchvision import transforms
from lighter.transform import BaseTransform


class Transform(BaseTransform):
    def __init__(self):
        super(Transform, self).__init__([
            transforms.ToTensor(),
        ])
