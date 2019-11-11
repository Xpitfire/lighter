import os
import json

import torch
import numpy as np
import pandas as pd
from PIL import Image

from lighter.dataset import BaseDataset
from lighter.decorator import config


class LookingDataset(BaseDataset):
    @config(path='minerl/datasets/defaults.config.json', group='dataset')
    def __init__(self, transform=None):
        super(LookingDataset, self).__init__()
        self.transform = transform
        self.root_dir = self.config.dataset.root_dir
        self.relative_dir = self.config.dataset.relative_dir
        self.source_file = self.config.dataset.source_file
        # TODO: download data if not available and build up self.data and self.target
        self.data = None
        self.target = None

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        # get index from tensor
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data_id = self.target.iloc[idx, 0]

        if data_id not in self.data:
            # if lazy loading then load image
            # TODO: load if not already available
            pass

        data = self.data[data_id]
        if self.transform:
            data = self.transform(data)

        # TODO: set data and target return type
        return None, None