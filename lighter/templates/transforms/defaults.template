from torchvision import transforms
from lighter.transform import BaseTransform


class Transform(BaseTransform):
    def __init__(self):
        super(Transform, self).__init__()
        self.caller = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __call__(self, values):
        return self.caller(values)
