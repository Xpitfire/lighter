from lighter.context import Context
from experiments.defaults import Experiment

if __name__ == '__main__':
    Context.create()
    experiment = Experiment()
    experiment.run()
