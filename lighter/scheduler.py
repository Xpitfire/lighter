import os
import json
import torch
from torch.multiprocessing import Process
from lighter.search import ParameterSearch
from lighter.context import Context
from lighter.loader import Loader
from lighter.config import Config
from lighter.registry import Registry


class ContextBuilder(object):
    def __init__(self,
                 process_id: str,
                 schedule_file: str,
                 experiment: str,
                 device: str = 'cpu'):
        self.process_id = process_id
        self.schedule_file = schedule_file
        self.device = device
        if 'cuda:' in self.device:
            torch.cuda.set_device(int(self.device.split(':')[-1]))
        with open(schedule_file) as json_file:
            schedule = json.load(json_file)
        self.files = schedule[self.process_id]
        self.experiment = Loader.import_path(experiment)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < len(self.files):
            context = Context.create(self.files[self.idx],
                                     parse_args_override=True,
                                     instantiate_types=False)

            context.config.device.default = self.device
            context.config.set_value('process_id', self.process_id)

            context.registry = Registry.create_instance()
            config = Config.create_instance(self.files[self.idx],
                                            parse_args_override=True)
            config.set_value('process_id', self.process_id)
            config.device.default = self.device
            context.config = config
            context.instantiate_types(context.registry.types)

            experiment = self.experiment()
            setattr(experiment, 'config', config)
            search = ParameterSearch.create_instance()
            setattr(experiment, 'search', search)
            for key in config['strategy'].keys():
                setattr(experiment, key, context.registry.instances[key])

            self.idx += 1
            return experiment
        else:
            raise StopIteration


class Executor:
    @staticmethod
    def worker(process_id: str, schedule_file: str, experiment: str, device):
        scheduler = ContextBuilder(process_id=process_id,
                                   schedule_file=schedule_file,
                                   experiment=experiment,
                                   device=device)
        for i, exp in enumerate(scheduler):
            exp()


class Scheduler:
    def __init__(self,
                 path: str,
                 experiment: str,
                 device_name: str = 'cuda',
                 num_workers: int = 1):
        self.path = path
        self.experiment = experiment
        self.device_name = device_name
        self.num_workers = num_workers
        self.schedule_file = None
        self.processes = []

    def build_schedule(self):
        # create empty schedule
        schedule = {'{}:{}'.format(self.device_name, i % self.num_workers): [] for i in range(self.num_workers)}
        # assign configs to schedule
        for i, file in enumerate(os.listdir(self.path)):
            schedule['{}:{}'.format(self.device_name, i % self.num_workers)].append(
                os.path.join(self.path, file))
        # save schedule
        self.schedule_file = os.path.join(self.path, 'schedule.json')
        dict_str = json.dumps(schedule, indent=2)
        with open(self.schedule_file, 'w') as file:
            file.write(dict_str)

    def create_processes(self):
        for i in range(self.num_workers):
            process_id = '{}:{}'.format(self.device_name, i)
            device = process_id
            if 'cpu' in device:
                device = 'cpu'
            process = Process(name=process_id,
                              target=Executor.worker,
                              args=(process_id, self.schedule_file, self.experiment, device))
            self.processes.append(process)

    def execute_processes(self):
        for t in self.processes:
            t.start()
        [p.join() for p in self.processes]

    def clean_processes(self):
        self.processes = []

    def run(self):
        self.build_schedule()
        self.create_processes()
        self.execute_processes()
        self.clean_processes()
