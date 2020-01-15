import os
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from lighter.decorator import device, context
from lighter.misc import generate_long_id


class BaseExperiment(object):
    """
    Experiment base class to create new algorithm runs.
    An experiment is also iterable and can be used as an iterator.
    It allows different hooks to interact in the main loop.
    """
    @device
    @context
    def __init__(self,
                 experiment_id: str = None,
                 epochs: int = 100,
                 enable_checkpoints: bool = True,
                 checkpoints_dir: str = 'runs/',
                 checkpoints_interval: int = 1):
        if experiment_id is None:
            experiment_id = generate_long_id()
        self.config['experiment_id'] = experiment_id
        self.epoch = 0
        self.epochs = epochs
        self.enable_checkpoints = enable_checkpoints
        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_interval = checkpoints_interval

    def __call__(self, *args, **kwargs):
        self.run()

    def __iter__(self):
        self.initialize()
        self.eval()
        return self

    def __next__(self):
        if self.epoch < self.epochs:
            self._epoch(self.epoch)
            self.epoch += 1
            return self.epoch
        else:
            self.finalize()
            raise StopIteration

    def _epoch(self, epoch):
        self.pre_epoch()
        self.train()
        self.eval()
        self.checkpoint(epoch)
        self.post_epoch()

    def run(self):
        """
        Main entrance point for an experiment.
        :return:
        """
        self.initialize()
        self.eval()
        for epoch in tqdm(range(self.epochs)):
            self._epoch(epoch)
        self.finalize()

    def initialize(self):
        """
        Initialize the experiment phase.
        :return:
        """
        self.epoch = 0
        # save the new experiment config
        path = os.path.join(self.checkpoints_dir, self.config.context_id, self.config.experiment_id)
        if not os.path.exists(path):
            os.makedirs(path)
        config_file = os.path.join(path, 'experiment.config.json')
        self.config.save(config_file)

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

    def finalize(self):
        """
        Post experiment cleanup code.
        :return:
        """
        pass


class DefaultExperiment(BaseExperiment):
    """
    Simple implementation of the experiment base class to execute train / eval runs.
    """
    @device
    @context
    def __init__(self,
                 experiment_id: str = None,
                 epochs: int = 100,
                 enable_checkpoints: bool = True,
                 checkpoints_dir: str = 'runs/',
                 checkpoints_interval: int = 1,
                 redux_function=np.mean):
        super(DefaultExperiment, self).__init__(experiment_id=experiment_id,
                                                epochs=epochs,
                                                enable_checkpoints=enable_checkpoints,
                                                checkpoints_dir=checkpoints_dir,
                                                checkpoints_interval=checkpoints_interval)
        self.train_loader, self.val_loader = None, None
        self.redux_function = redux_function

    def initialize(self):
        # get data loaders
        self.train_loader, self.val_loader = self.data_builder.loader()
        super().initialize()

    def pre_epoch(self):
        pass

    def post_epoch(self):
        self.writer.step()
        self.collectible.reset()

    def train_batch(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = self.criterion(pred, y)
        loss.backward()
        self.optimizer.step()
        self.collectible.update(category='train', **{'loss': loss.detach().cpu().item()})
        self.collectible.update(category='train', **self.metric(pred.detach().cpu(),
                                                                y.detach().cpu()))

    def train(self):
        if self.train_loader is not None:
            self.model.train()
            for i, (x, y) in enumerate(self.train_loader):
                self.train_batch(x, y)
            collection = self.collectible.redux(func=self.redux_function)
            self.writer.write(category='train', **collection)

    def eval_batch(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        pred = self.model(x)
        loss = self.criterion(pred, y)
        self.collectible.update(category='eval', **{'loss': loss.detach().cpu().item()})
        self.collectible.update(category='eval', **self.metric(pred.detach().cpu(),
                                                               y.detach().cpu()))

    def eval(self):
        if self.val_loader is not None:
            self.model.eval()
            with torch.no_grad():
                for i, (x, y) in enumerate(self.val_loader):
                    self.eval_batch(x, y)
                collection = self.collectible.redux(func=self.redux_function)
                self.writer.write(category='eval', **collection)

    def checkpoint(self, epoch: int):
        if self.enable_checkpoints and epoch % self.checkpoints_interval == 0:
            collection = self.collectible.redux(func=self.redux_function)
            timestamp = datetime.timestamp(datetime.now())
            file_name = 'e-{}_time-{}'.format(epoch, timestamp)
            path = os.path.join(self.checkpoints_dir, self.config.context_id, self.config.experiment_id)
            ckpt_file = os.path.join(path, '{}.ckpt'.format(file_name))
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': collection
            }, ckpt_file)
