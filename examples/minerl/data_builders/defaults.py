import torch
import numpy as np
from torch.utils.data import SubsetRandomSampler

from lighter.data_builder import BaseDataBuilder
from lighter.decorator import config


class DataBuilder(BaseDataBuilder):
    @config(path='data_builders/defaults.config.json', property='data_builder')
    def __init__(self):
        super(DataBuilder, self).__init__()
    
    def loader(self):
        batch_size = self.config.data_builder.batch_size
        validation_split = self.config.data_builder.validation_split
        shuffle_dataset = self.config.data_builder.shuffle_dataset
        random_seed = self.config.data_builder.random_seed

        # Creating data indices for training and validation splits:
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(self.dataset,
                                                   batch_size=batch_size,
                                                   sampler=train_sampler,
                                                   num_workers=4)
        validation_loader = torch.utils.data.DataLoader(self.dataset,
                                                        batch_size=batch_size,
                                                        sampler=valid_sampler,
                                                        num_workers=4)

        return train_loader, validation_loader
