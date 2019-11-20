import os
from multiprocess import Process
from lighter.search import ParameterSearch
from lighter.context import Context
from lighter.loader import Loader
from lighter.config import Config
from lighter.registry import Registry
from multiprocessing import set_start_method


class ContextBuilder(object):
    def __init__(self, path: str, experiment: str, device_name: str = 'cuda', num_worker: int = 1):
        self.device_name = device_name
        self.num_worker = num_worker
        self.files = [os.path.join(path, file) for file in os.listdir(path)]
        self.experiment = Loader.import_path(experiment)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < len(self.files):
            context = Context.create(self.files[self.idx],
                                     parse_args_override=True,
                                     instantiate_types=False)

            device = '{}:{}'.format(self.device_name, self.idx % self.num_worker)
            process_id = device
            context.config.set_value('process_id', process_id)
            if 'cpu' == self.device_name:
                device = 'cpu'

            context.config.device.default = device
            experiment = self.experiment()

            context.registry = Registry.create_instance()
            config = Config.create_instance(self.files[self.idx],
                                            parse_args_override=True)
            config.set_value('process_id', process_id)
            config.device.default = device
            context.config = config
            context.instantiate_types(context.registry.types)

            setattr(experiment, 'config', config)
            search = ParameterSearch.create_instance()
            setattr(experiment, 'search', search)
            for key in config['strategy'].keys():
                setattr(experiment, key, context.registry.instances[key])

            self.idx += 1
            return experiment
        else:
            raise StopIteration


class Runner:
    def __init__(self, tasks):
        self.tasks = tasks

    def run(self):
        for task in self.tasks:
            task()


class Executor:
    def __init__(self, device_name, pool):
        self.device_name = device_name
        self.pool = pool

        self._cuda_fix()
        self.processes = []
        for process_id, tasks in pool.items():
            process = Process(name=process_id,
                              target=Executor.worker,
                              args=tasks)
            self.processes.append(process)

    def _cuda_fix(self):
        if 'cuda' in self.device_name:
            # used for cuda multiprocessing
            try:
                set_start_method('spawn')
            except RuntimeError:
                pass

    @staticmethod
    def worker(*args):
        print(args)
        runner = Runner(args)
        runner.run()

    def start(self):
        for t in self.processes:
            t.start()
        [p.join() for p in self.processes]


class Scheduler:
    def __init__(self,
                 path: str,
                 experiment: str,
                 device_name: str = 'cuda',
                 num_worker: int = 1):
        self.path = path
        self.experiment = experiment
        self.device_name = device_name
        self.num_worker = num_worker
        self.scheduler = ContextBuilder(path=path,
                                        experiment=experiment,
                                        device_name=device_name,
                                        num_worker=num_worker)

    def run(self):
        pool = {}
        for i, exp in enumerate(self.scheduler):
            process_id = exp.config.process_id
            if process_id not in pool.keys():
                pool[process_id] = []
            pool[process_id].append(exp)
            if i > self.num_worker:  # TODO: remove after debugging
                break
        executor = Executor(self.device_name, pool)
        executor.start()
