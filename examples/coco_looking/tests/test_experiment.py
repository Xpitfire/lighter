from lighter.context import Context
from experiments.defaults import SimpleExperiment

if __name__ == '__main__':
    Context.create()
    experiment = SimpleExperiment()
    experiment.run()
