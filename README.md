# Lighter

Lightweight extension for torch to speed up project prototyping and enable dependency injection of object instances.
This framework includes a setup script to get started with a pre-defined project structure and the documentation offers examples and an overview of the initial usage.

## Install

```shell script
$> pip install torch-lighter
```

## Quickstart

Create a JSON file `config.json`:

```json
{
  "text": "hello world"
}
```

Create a python file and use the decorators to inject the config instance:

```python
class Demo:
    @config(path='config.json', property='config')
    def __init__(self):
        pass
    
    def run(self):
        print(self.config.text)

# creates the application context
Context.create()

# creates the object instance and injects the configs
Demo().run()
```

This will print the following output:

```shell script
[]: hello world
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
  * __init__.py
  * collectibles
    - __init__.py
    - defaults.py
  * configs
    - modules.config.json
  * criterions
    - __init__.py
    - defaults.config.json
    - defaults.py
  * data_buildres
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
  * metrics
    - __init__.py
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
  * writers
    - __init__.py
    - defaults.config.json
    - defaults.py
```

Of course this is only a preset example and the projects can be structured at will.

### Structure description
- collectibles: are used to gather metrics over multiple steps / epochs
- criterions: torch criterion classes
- data_builders: data builder classes creating the data loaders
- datasets: dataset used by the data builder
- experiments: experiment logic files containing the main `run()`, `train()` and `eval()` methods
- metrics: metrics used to measure performance
- modules: references all available modules
- optimizers: optimizers used for performing the step
- tests: contains the main testing sources for the project and mark the main entrance point
- transforms: data transforms are used by the datasets to reshape the data structures
- writers: tensorboard writers

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

## Advanced

### Inheritance from base class

One can simply extend from any base class within the lighter framework:
```python
class SimpleExperiment(BaseExperiment):
    @config(path='experiments/defaults.config.json', property='experiments')
    def __init__(self):
        super(SimpleExperiment, self).__init__()

    def eval(self):
        ...

    def train(self):
        ...
        preds = self.model(inputs)
        loss = self.metric(preds, targets)
        ...

    def run(self):
        for epoch in self.config.experiments.num_epochs:
            self.train()
            self.eval()
```
In this example we use the `BaseExperiment` class and create a new experiment. 
The `@config` decorator injects the `defaults.config.json` into the `self.config` instance in a sub-property `experiments` of the default config.
One can then access the properties by de-referencing `self.config.experiments.<property>`.
`self.model` and `self.metric` are injected by the base class and are usually referenced in the `modules/defaults.config.json`.

### Decorators

It is also possible to simply use the required decorators without base class sub-classing. 
There are currently five decorators:
* `@config` - injects the default config instance and allows to load new configs
* `@context` - injects the application context which holds config and registry instances
* `@dataset` - injects dataset instance
* `@model` - injects model instance
* `@reference` - injects a set of pre-set or defined instances
* `@strategy` - loads and defines a training set strategy from a config and injects the main object instances
* `@inject` - allows to inject single instances from different context options (registry, types, instances, configs)
* `@register` - allows to quickly register a new type to the context registry and inject the instance into the current instance

By default the templates for a new project are using these names for dependency injection `['dataset', 'data_builder', 'criterion', 'model', 'writer', 'optimizer', 'transfom', 'metric', 'collectible']` and are injecting the instances into the `experiments.Experiment` instance.
```json
{
    "modules": {
        "transform": "type::datasets.defaults.Transform",
        "dataset": "type::datasets.defaults.Dataset",
        "data_builder": "type::data_builders.defaults.DataBuilder",
        "criterion": "type::criterions.defaults.Criterion",
        "model": "type::models.defaults.Model",
        "writer": "type::writers.defaults.Writer",
        "optimizer": "type::optimizers.defaults.Optimizer",
        "metric": "type::metrics.defaults.Metric",
        "collectible": "type::collectibles.defaults.Collectible"
    }
}
```

But it is simple to inject custom objects or exchangeable names by referring to a new name or type:
```json
{
    "modules": {
        ...
        "other_model_name": "type::models.defaults.Network",
        ...
    }
}
```

This now allows to load a model using the `@model` decorator withe the new property reference.

```python
class Demo:
    @model(properties=['other_model_name'])
    def __init__(self):
        ...
        self.other_model_name.<property>
        ...
```

### Config templates

All configs are based on `json` files. Yet, when importing the configs `lighter` parses some escape sequences allowing for more modularization of your project structure:
* `config::<config-file-path>` - imports another config into the current json config instance
* `import::<config-file-path>` - imports all properties of the referenced config file into the current config instance (attention may override existing properties)
* `type::<type-file-path>` - imports a python type which is registered to the application context `registry.types` and can be used to instantiate new objects

