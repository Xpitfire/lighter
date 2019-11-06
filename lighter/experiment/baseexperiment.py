import tqdm
import os
import numpy as np
from datetime import datetime
from lighter.config import get_default_config
from torch.utils.tensorboard import SummaryWriter


class BaseExperiment(object):
    """
    Experiment base class to create new algorithm runs.
    """
    def __init__(self):
        self.config = get_default_config()

    def __call__(self, *args, **kwargs):
        pass


class SimpleExperiment(BaseExperiment):
    def __init__(self):
        super(SimpleExperiment, self).__init__()
        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.metric = None
        self.writer = None
        self.collectible = None

        self.steps = 0
        if self.config.enable_logs:
            self.writer = SummaryWriter(log_dir=self.config.log_dir)

    def train(self):
        self.model.train()
        for i, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.config.device), y.to(self.config.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            self.write(category='train', **{'loss': loss.cpu()})
            self.write(category='train', **self.metric(y.cpu(), pred.cpu()))
            self.steps += 1

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(self.val_loader):
                x, y = x.to(self.config.device), y.to(self.config.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                self.collectible.update(**{'loss': loss.cpu()})
                self.collectible.update(**self.metric(y.cpu(), pred.cpu()))
            collection = self.collectible.redux(func=np.mean)
            self.write(category='eval', **collection)

    def checkpoint(self, epoch):
        if self.config.enable_ckpts:
            collection = self.collectible.redux(func=np.mean)
            timestamp = datetime.timestamp(datetime.now())
            ckpt_file = os.path.join(self.config.ckpt_dir, 'e-{}_time-{}.ckpt'.format(epoch, timestamp))
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': collection['loss'],
                'val_acc': collection['acc']
            }, ckpt_file)

    def run(self):
        self.train_loader, self.val_loader = dl.build_loader(
            root_dir=self.config.data_dir,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split
        )
        self.model = self.get_model()
        self.optimizer = self.get_optimizer()
        self.criterion = self.get_criterion()
        self.metric = self.get_metric()
        self.collectible = self.get_collectible()

        for epoch in tqdm(range(self.config.epochs)):
            self.train()
            self.evaluate()
            self.checkpoint(epoch)
            self.collectible.reset()
