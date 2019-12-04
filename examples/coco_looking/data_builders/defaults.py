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
        shuffle_sampler = self.config.data_builder.shuffle_sampler
        random_seed = None
        if hasattr(self.config.data_builder, 'random_seed'):
            random_seed = self.config.data_builder.random_seed
        num_workers = self.config.data_builder.num_workers
        # Creating data indices for training and validation splits:
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if random_seed:
            np.random.seed(random_seed)
        if shuffle_sampler:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(self.dataset,
                                                   batch_size=batch_size,
                                                   sampler=train_sampler,
                                                   num_workers=num_workers)
        validation_loader = torch.utils.data.DataLoader(self.dataset,
                                                        batch_size=batch_size,
                                                        sampler=valid_sampler,
                                                        num_workers=num_workers)

        return train_loader, validation_loader
