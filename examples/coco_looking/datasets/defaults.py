import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from lighter.dataset import BaseDataset
from lighter.decorator import config
from lighter.utils.inet import download_and_extract_zip


class LookingDataset(BaseDataset):
    @config(path='datasets/defaults.config.json', property='dataset')
    def __init__(self):
        super(LookingDataset, self).__init__()
        self.root_dir = self.config.dataset.root_dir
        self.source_file = self.config.dataset.source_file
        json_file = os.path.join(self.root_dir, self.source_file)
        if not os.path.exists(json_file) and self.config.dataset.download_data:
            download_and_extract_zip(self.config.dataset.data_url, self.config.dataset.root_dir)
        if not os.path.exists(json_file):
            raise FileNotFoundError('Looking Dataset is missing the meta file: {}'.format(json_file))
        with open(json_file, 'r') as file:
            looking_dict = json.load(file)
        self.data = {}
        self.target = []
        for k, v in looking_dict.items():
            self.target.append((k, v))
        self.target = pd.DataFrame(self.target)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        # get index from tensor
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.target.iloc[idx, 0]

        if img_name not in self.data:
            # if lazy loading then load image
            file = os.path.join(self.root_dir, img_name)
            image = Image.open(file)
            # copy as a PIL Image issue workaround
            self.data[img_name] = image.copy()
            image.close()

        data = self.data[img_name]
        if self.transform:
            data = self.transform(data)
        return data, np.array([self.target.iloc[idx, 1]]).astype(np.float32)
