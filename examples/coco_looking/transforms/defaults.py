from torchvision import transforms
from lighter.transform import BaseTransform


class Transform(BaseTransform):
    def __init__(self):
        super(Transform, self).__init__()
        self.caller = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # set imagenet data normalization
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, values):
        return self.caller(values)
