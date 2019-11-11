import os
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

from lighter.decorator import config
from lighter.experiment import BaseExperiment


class SimpleExperiment(BaseExperiment):
    @config(path='minerl/experiments/defaults.config.json', group='experiment')
    def __init__(self):
        super(SimpleExperiment, self).__init__()
        self.train_loader, self.val_loader = self.data_builder.loader()

    def train(self):
        self.model.train()
        for i, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.config.model.device), y.to(self.config.model.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            self.writer.write(category='train', **{'loss': loss.cpu()})
            self.writer.write(category='train', **self.metric(y.cpu(), pred.cpu()))
            self.writer.step()

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(self.val_loader):
                x, y = x.to(self.config.model.device), y.to(self.config.model.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                self.collectible.update(**{'loss': loss.cpu()})
                self.collectible.update(**self.metric(y.cpu(), pred.cpu()))
            collection = self.collectible.redux(func=np.mean)
            self.writer.write(category='eval', **collection)

    def checkpoint(self, epoch):
        if self.config.experiment.enable_ckpts:
            collection = self.collectible.redux(func=np.mean)
            timestamp = datetime.timestamp(datetime.now())
            ckpt_file = os.path.join(self.config.experiment.ckpt_dir, 'e-{}_time-{}.ckpt'.format(epoch, timestamp))
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': collection['loss'],
                'val_acc': collection['acc']
            }, ckpt_file)

    def run(self, *args, **kwargs):
        for epoch in tqdm(range(self.config.experiment.epochs)):
            self.train()
            self.eval()
            self.checkpoint(epoch)
            self.collectible.reset()