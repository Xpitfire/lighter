import os
import torch
import numpy as np

from tqdm import tqdm
from datetime import datetime
from lighter.decorator import device, references, context


class BaseExperiment(object):
    """
    Experiment base class to create new algorithm runs.
    An experiment is also iterable and can be used as an iterator.
    It allows different hooks to interact in the main loop.
    """
    @device
    @context
    def __init__(self,
                 epochs: int = 100,
                 enable_checkpoints: bool = True,
                 checkpoints_dir: str = 'runs/',
                 checkpoints_interval: int = 1):
        self.epochs = epochs
        self.enable_checkpoints = enable_checkpoints
        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_interval = checkpoints_interval

    def __call__(self, *args, **kwargs):
        self.run()

    def __iter__(self):
        self.epoch = 0
        # save the new experiment config
        path = os.path.join(self.checkpoints_dir, self.config.context_id)
        config_file = os.path.join(path, 'experiment.config.json')
        self.config.save(config_file)
        return self

    def __next__(self):
        if self.epoch < self.epochs:
            self.pre_epoch()
            self.train()
            self.eval()
            self.checkpoint(self.epoch)
            self.post_epoch()
            self.epoch += 1
            return self.epoch
        else:
            raise StopIteration

    def run(self):
        """
        Main entrance point for an experiment.
        :return:
        """
        for epoch in tqdm(range(self.epochs)):
            self.pre_epoch()
            self.train()
            self.eval()
            self.checkpoint(epoch)
            self.post_epoch()

    def pre_epoch(self):
        """
        Hook that can be overridden before a training epoch starts.
        :return:
        """
        raise NotImplementedError('BaseExperiment: No implementation found!')

    def post_epoch(self):
        """
        Hock that can be overridden after training epoch ended.
        The current implementation resets the collectible and steps the writer.
        :return:
        """
        raise NotImplementedError('BaseExperiment: No implementation found!')

    def train(self):
        """
        Training instance for the experiment run.
        :return:
        """
        raise NotImplementedError('BaseExperiment: No implementation found!')

    def eval(self):
        """
        Evaluates an experiment.
        :return:
        """
        raise NotImplementedError('BaseExperiment: No implementation found!')

    def checkpoint(self, epoch: int):
        """
        Code for model state save.
        :param epoch: Current epoch executed.
        :return:
        """
        raise NotImplementedError('BaseExperiment: No implementation found!')


class DefaultExperiment(BaseExperiment):
    """
    Simple implementation of the experiment base class to execute train / eval runs.
    """
    @device
    @context
    @references
    def __init__(self,
                 epochs: int = 100,
                 enable_checkpoints: bool = True,
                 checkpoints_dir: str = 'runs/',
                 checkpoints_interval: int = 1):
        super(DefaultExperiment, self).__init__(epochs=epochs,
                                                enable_checkpoints=enable_checkpoints,
                                                checkpoints_dir=checkpoints_dir,
                                                checkpoints_interval=checkpoints_interval)
        # get data loaders
        self.train_loader, self.val_loader = self.data_builder.loader()

    def pre_epoch(self):
        pass

    def post_epoch(self):
        self.writer.step()
        self.collectible.reset()

    def train(self):
        if self.train_loader is not None:
            self.model.train()
            for i, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                self.collectible.update(category='train', **{'loss': loss.detach().cpu().item()})
                self.collectible.update(category='train', **self.metric(y.detach().cpu(),
                                                                        pred.detach().cpu()))
            collection = self.collectible.redux(func=np.mean)
            self.writer.write(category='train', **collection)

    def eval(self):
        if self.val_loader is not None:
            self.model.eval()
            with torch.no_grad():
                for i, (x, y) in enumerate(self.val_loader):
                    x, y = x.to(self.device), y.to(self.device)
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                    self.collectible.update(category='val', **{'loss': loss.detach().cpu().item()})
                    self.collectible.update(category='val', **self.metric(y.detach().cpu(),
                                                                          pred.detach().cpu()))
                collection = self.collectible.redux(func=np.mean)
                self.writer.write(category='eval', **collection)

    def checkpoint(self, epoch: int):
        if self.enable_checkpoints and epoch % self.checkpoints_interval == 0:
            collection = self.collectible.redux(func=np.mean)
            timestamp = datetime.timestamp(datetime.now())
            file_name = 'e-{}_time-{}'.format(epoch, timestamp)
            path = os.path.join(self.checkpoints_dir, self.config.context_id)
            if not os.path.exists(path):
                os.makedirs(path)
            ckpt_file = os.path.join(path, '{}.ckpt'.format(file_name))
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': collection
            }, ckpt_file)
