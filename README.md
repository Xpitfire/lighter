# Lighter

Lightweight extension for torch to speed up project prototyping and enable dependency injection of object instances.
This framework includes a setup script to get started with a pre-defined project structure and the documentation offers examples and an overview of the initial usage.
For more examples follow the [link to the GitHub repository](https://github.com/Xpitfire/lighter).

## Install

```shell script
$> pip install torch-lighter
```

## Quickstart

Create a JSON file `config.json`:

```json
{
  "num_of_epochs": 50
}
```

Create a python file and use the decorators to inject the config instance:

```python
# creates the application context
Context.create()

class Demo:
    @config(path='config.json', property='config')
    def __init__(self):
        pass
    
    def run(self):
        print(self.config.num_of_epochs)

# creates the object instance and injects the configs
Demo().run()
```

This will print the following output:

```shell script
[]: 50
```

Lighter can be simply added to any existing project or one can create a new lighter template project as shown in the next section.


## Create a project

### Initialize project

Type the following in your command line / bash:
```shell script
$> lighter-init <project-name>
``` 

### Structure overview
The above command will create the following project structure:
```text
my_project:
  * configs
    - modules.config.json
  * criterions
    - __init__.py
    - defaults.config.json
    - defaults.py
  * data_builders
    - __init__.py
    - defaults.config.json
    - defaults.py
  * datasets
    - __init__.py
    - defaults.config.json
    - defaults.py
  * experiments
    - __init__.py
    - defaults.config.json
    - defaults.py
  * models
    - __init__.py
    - defaults.config.json
    - defaults.py
  * optimizers
    - __init__.py
    - defaults.config.json
    - defaults.py
  * tests
    - __init__.py
    - test_experiment.py
  * transforms
    - __init__.py
    - defaults.py
  * __init__.py
  * README.md
  * LICENSE
  * setup.py
```

Of course this is only a preset example and the projects can be structured at will.

### Modules description
- collectible: are used to gather metrics over multiple steps / epochs
- criterion: torch criterion classes
- data_builder: data builder classes creating the data loaders
- dataset: dataset used by the data builder
- experiment: experiment logic files containing the main `run()`, `train()` and `eval()` methods
- metric: metrics used to measure performance
- module: references all available modules
- optimizer: optimizers used for performing the step
- test: contains the main testing sources for the project and mark the main entrance point
- transform: data transforms are used by the datasets to reshape the data structures
- writer: tensorboard writers

### TODOs

Each directory containing a python file may also contain a config file outsourcing properties for testing.
Modules that require further code insertion are marked with `TODO` flags.

### Run an experiment

After creating a demo project and inserting code into the `TODO` placeholders simply run an experiment by calling:
```python
$> python tests/test_experiment.py
```

It might be necessary to set your `PYTHONPATH` before running the experiment file:

```python
$> PYTHONPATH=. python tests/test_experiment.py
```

One can also simply change the global device setting or initialize an experiment from command line by specifying theses arguments:
```python
$> python tests/test_experiment.py --device cuda:1 --config <path-to-config>
```

## Advanced

### Inheritance from base class

One can simply extend from any base class within the lighter framework:
```python
class SimpleExperiment(DefaultExperiment):
    @config(path='experiments/defaults.config.json', property='experiments')
    @strategy(config='configs/modules.config.json')
    @references
    def __init__(self):
        super(SimpleExperiment, self).__init__(epochs=self.config.experiments.num_epochs)

    def eval(self):
        ...

    def train(self):
        ...
        preds = self.model(inputs)
        loss = self.metric(preds, targets)
        ...
```
In this example we subclass the `DefaultExperiment` class and create a new experiment with some custom `eval()` and `train()` behaviour. 
The `run()` work-flow is still used as defined in the base class `DefaultExperiment`.
The `@references` decorator injects by default some properties into the current `SimpleExperiment` object instance. 
If not other specified the `@reference` tries to inject the properties `['dataset', 'data_builder', 'criterion', 'model', 'writer', 'optimizer', 'transfom', 'metric', 'collectible']` which in this case are declared in the `configs/modules.defaults.config.json` config.
Since `model` and `metric` are injected via `@references` they can be directly accessed via the `self` instance.


The `@config` decorator injects additional configs for the current experiment instance by referencing `defaults.config.json` with an `experiments` alias. 
All configs are by default created as a sub-property of `self.config` and are now referencable through the `experiments` alias.
One can then access the properties as follows `self.config.experiments.<property>`.

### Decorators

In general it is also possible to simply use the required decorators without base class sub-classing.

 
There are currently the following decorators available:
* `@config` - injects the default config instance and allows to load new configs
* `@context` - injects the application context which holds config and registry instances
* `@transform` - injects the default data transform instance
* `@dataset` - injects the default dataset instance
* `@model` - injects the default model instance
* `@metric` - injects the default metric instance
* `@reference` - injects a singe defined instance
* `@references` - injects a set of pre-set or defined instance(s)
* `@strategy` - defines and loads a training strategy from a specified config path and injects the default object instances
* `@register` - allows to quickly register a new type to the context registry and inject the instance into the current instance
* `@hook` - allows to hook and overwrite an existing method to change the execution logic
* `@inject` - allows to inject single instances into different context options (registry, types, instances, configs)
* `@search` - allows to register hyper-parameters to parse a config schedule for parallelled executions

By default the templates for a new project are creating a config file in the `configs/modules.config.json` path, which uses the following default names for the dependency injection `['dataset', 'data_builder', 'criterion', 'model', 'writer', 'optimizer', 'transfom', 'metric', 'collectible']`.
An experiment gets these instances automatically instantiated and injected at runtime.
```json
{
  "strategy": {
      "transform": "type::transforms.defaults.Transform",
      "dataset": "type::datasets.defaults.Dataset",
      "data_builder": "type::data_builders.defaults.DataBuilder",
      "criterion": "type::criterions.defaults.Criterion",
      "model": "type::models.defaults.Model",
      "optimizer": "type::optimizers.defaults.Optimizer",
      "metric": "type::lighter.metric.BaseMetric",
      "collectible": "type::lighter.collectible.BaseCollectible",
      "writer": "type::lighter.writer.BaseWriter"
  }
}
```

The above example groups the types in the `strategy` alias name to avoid config name collisions if other classes import configs using the same general names (model, metric, etc.) across the application.

It is also simple to deviate from this naming convention or even inject custom types by exchanging names within the referencing configs:
```json
{
  ...
  "other_model": "type::models.other.Network",
  "other_metric": "type::metrics.other.Metric",
  ...
}
```

The defined name `other_model_name` is now uniquely assigned in the instance context and can be accessed via `@references(names=['other_model', 'other_metric'])`

It is also possible to register types on the fly within the code at the class or method level:

```python
class Demo:
    @register(type='envs.defaults.Environment', property='env')
    @reference(name='env')
    def __init__(self):
        ...
        obs = self.env.reset()
        ...
```

Here the `@reference` decorator requires a name argument to match the instance.

### Config templates

All configs are based on `json` files. Yet, when importing the configs `lighter` parses some escape sequences allowing for more modularization of your project structure:

* `config::<config-file-path>` - imports another config into the current json config instance
* `import::<config-file-path>` - imports all properties of the referenced config file into the current config instance (attention may override existing properties)
* `type::<type-file-path>` - imports a python type which is registered to the application context `registry.types` and can be used to instantiate new objects

### Parameter search parser

One can simply use lighter for searching hyper-parameters or different setting using the `@search` decorator, which registers parameters referenced by configs to the global context.
This generates a permutation of different settings using the parameter parser, which then allows for parallel execution either with the default built in scheduler or any other specialized framework for distributed optimization, such as [ray](https://github.com/ray-project/ray).

```python
class ParameterSearchRegistration:
  @search(group='sgd',
          params=[('lr', GridParameter(ref='optimizer.lr', min=0.001, max=0.005, step=0.001)),
                  ('weight_decay', ListParameter(ref='optimizer.weight_decay', options=[0.0, 0.9])),
                  ('freeze_pretrained', BinaryParameter(ref='model.freeze_pretrained')),
                  ('hidden_units', GridParameter(ref='model.hidden_units', min=100, max=200, step=100)),
                  ('optimizer', SetParameter(ref='strategy.optimizer', option="type::optimizers.defaults.Optimizer")),
                  ('model_output', SetParameter(ref='model.output', option=1)),
                  ('model_pretrained', SetParameter(ref='model.pretrained', option=True)),
                  ('strategy', StrategyParameter(ref='strategy', options=['searches/coco_looking.config.json'])),
                  ('model', ListParameter(ref='strategy.model',
                                          options=['type::models.alexnet.AlexNetFeatureExtractionModel',
                                                   'type::models.resnet.ResNetFeatureExtractionModel']))])
  def __init__(self):
    pass
```

Here `group` marks the grouping alias under which the upcoming parameters are registered in the context.
The `params` argument takes in a list of two-tuples `(<name: str>, <param: Parameter>)`, where `name` marks the parameter alias within the group and `param` the type of search parameter used.

Currently lighter offers the following parameter types:

* `GridParameter`: Search a range of values (min, max, steps) with fixed step size
* `ListParameter`: Commit a list of parameters to lighter which is sequentially executed
* `CallableGridParameter`: Search a range of values (min, max, callable) but with a callable lambda function which allows for non-linear step behaviour
* `AnnealParameter`: Same as `CallableGridParameter` but for annealing a parameter
* `BinaryParameter`: Return `True` and `False` values for a parameter
* `StrategyParameter`: Allows to define different training strategies and switch between them
* `SetParameter`: Represents a simple setter of properties that is used to register settings, which don't change, but need to be imported

To trigger the parameter config parsing run:

```python
from searches.defaults import ParameterSearchRegistration
from lighter.parser import ConfigParser
from lighter.context import Context

context = Context.create(auto_instantiate_types=False)
psr = ParameterSearchRegistration()
cp = ConfigParser(experiment=psr)
cp.parse()
```

By default this saves all permuted configs to the `runs/search/<gen-name>` directory.
The `auto_instantiate_types` parameter prevents that the context instantiates the registered types when loading in configs.

### Parameter search scheduler

Lighter comes with a built in config scheduler that can execute experiments in parallel for single machine experiments.
For many prototyping cases this is sufficient and helps during development, but naturally flexibility is highest priority and it is possible and recommended to use 3rd party frameworks to schedule decentralized parallel executions on clusters of machines.

For the simple case, 

```python
import torch
from lighter.scheduler import Scheduler

# required to execute torch in parallel processes
torch.multiprocessing.set_start_method('spawn', force=True)
num_devices = torch.cuda.device_count()
# sets the path to the list of configs
scheduler = Scheduler(path='runs/search/witty-poodle',
                      experiment='experiments.defaults.Experiment',
                      device_name='cuda',
                      num_workers=num_devices)
scheduler.run()
```

The scheduler in the code snipped above starts as many processes as there are GPUs available and schedules the list of configs across these devices.
Since we defined a simple writer, one can simply monitor the progress using tensorboard.