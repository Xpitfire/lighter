import petname
from lighter.context import Context
from experiments.defaults import SimpleExperiment

if __name__ == '__main__':
    Context.create()
    generated_name = petname.Generate(2, '-', 6)
    Context.get_instance().config.set_value('experiment_name', generated_name)
    experiment = SimpleExperiment()
    experiment.run()
