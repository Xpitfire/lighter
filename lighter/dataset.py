from torch.utils.data import Dataset
from lighter.decorator import transform


class BaseDataset(Dataset):
    """
    Dataset base class preparing the data.
    """
    @transform
    def __init__(self):
        pass
