import torch
from lighter.scheduler import Scheduler

if __name__ == '__main__':
    # required to execute torch in parallel processes
    torch.multiprocessing.set_start_method('spawn', force=True)
    num_devices = torch.cuda.device_count()
    scheduler = Scheduler(path='runs/search/hip-osprey',
                          experiment='experiments.defaults.Experiment',
                          device_name='cuda',
                          num_workers=num_devices)
    scheduler.run()
