import torch
from lighter.scheduler import Scheduler

if __name__ == '__main__':
    # required to execute torch in parallel processes
    torch.multiprocessing.set_start_method('spawn', force=True)
    num_devices = 1#torch.cuda.device_count()
    scheduler = Scheduler(path='runs/search/golden-bird',
                          experiment='experiments.multiprocess.SearchExperiment',
                          device_name='cpu',
                          num_workers=num_devices)
    scheduler.run()
