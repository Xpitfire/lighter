import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from lighter.dataset import BaseDataset


class LookingDataset(BaseDataset):
    def __init__(self):
        super(LookingDataset, self).__init__()
        self.root_dir = "data/extract/"
        self.source_file = "looking_labels.json"
        json_file = os.path.join(self.root_dir, self.source_file)
        if not os.path.exists(json_file):
            BaseDataset.download_zip("https://www.dinu.at/wp-content/uploads/2019/11/COCO_Looking_Labels.zip",
                                     self.root_dir)
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
