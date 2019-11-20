import torch
from lighter.scheduler import Scheduler


def worker(*args):
    print(args)


if __name__ == '__main__':
    num_devices = torch.cuda.device_count()
    scheduler = Scheduler(path='runs/search/witty-tomcat',
                          experiment='lighter.experiment.DefaultExperiment',
                          device_name='cuda',
                          num_worker=num_devices)
    scheduler.run()

