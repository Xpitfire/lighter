import multiprocessing
import torch
from lighter.scheduler import Scheduler


def worker(*args):
    print(args)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    torch.multiprocessing.set_start_method('spawn', force=True)
    num_devices = 1#torch.cuda.device_count()
    scheduler = Scheduler(path='runs/search/solid-sloth',
                          experiment='experiments.multiprocess.SearchExperiment',
                          device_name='cuda',
                          num_workers=num_devices)
    scheduler.run()
