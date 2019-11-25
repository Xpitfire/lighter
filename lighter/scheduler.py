import os
import json
import torch
from torch.multiprocessing import Process
from lighter.context import Context
from lighter.loader import Loader


class ContextBuilder(object):
    """
    Is used to create a proper context object for running an experiment with the current
    """
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
        """
        Iterates over the schedules list of config files and prepares the context for execution.
        :return: a prepared experiment
        """
        if self.idx < len(self.files):
            # create new context with default config
            context = Context.create(config_file=self.files[self.idx],
                                     parse_args_override=True,
                                     auto_instantiate_types=False,
                                     allow_context_changes=False)

            # assign a new default device
            context.config.device.default = self.device
            # assign the process_id
            context.config.set_value('process_id', self.process_id)

            # create types
            context.instantiate_types(context.registry.types)

            # create a new experiment
            experiment = self.experiment()

            self.idx += 1
            return experiment
        else:
            raise StopIteration


class Executor:
    """
    Defines the main worker for executing a schedule flow.
    """
    @staticmethod
    def worker(process_id: str, schedule_file: str, experiment: str, device):
        scheduler = ContextBuilder(process_id=process_id,
                                   schedule_file=schedule_file,
                                   experiment=experiment,
                                   device=device)
        for i, exp in enumerate(scheduler):
            exp()


class Scheduler:
    """
    Schedules a list of parameters within a designated path for execution on multiple devices.
    """
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
        """
        Builds a schedule file from the defined path with configs.
        :return:
        """
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
        """
        Creates a list of processes with execution workers.
        :return:
        """
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
        """
        Executes a schedule on the initiated process workers.
        :return:
        """
        for t in self.processes:
            t.start()
        [p.join() for p in self.processes]

    def clean_processes(self):
        """
        Deletes the list of processes.
        :return:
        """
        self.processes = []

    def run(self):
        """
        Main execution loop for starting a scheduled run.
        :return:
        """
        self.build_schedule()
        self.create_processes()
        self.execute_processes()
        self.clean_processes()
