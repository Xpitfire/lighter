import torch
from lighter.scheduler import Scheduler


def worker(*args):
    print(args)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    num_devices = 1#torch.cuda.device_count()
    scheduler = Scheduler(path='runs/search/witty-poodle',
                          experiment='experiments.multiprocess.SearchExperiment',
                          device_name='cpu',
                          num_workers=num_devices)
    scheduler.run()
