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
$> PYTHONPATH=. python tests/test_experiment.py --device cuda:1 --config <path-to-config>
```

## Advanced

### Inheritance from base class

One can simply extend from any base class within the lighter framework:
```python
class SimpleExperiment(DefaultExperiment):
    @config(path='experiments/defaults.config.json', property='experiments')
    def __init__(self):
        super(SimpleExperiment, self).__init__(epochs=self.config.experiments.num_epochs)

    def eval(self):
        ...

    def train(self):
        ...
        preds = self.model(inputs)
        loss = self.metric(preds, targets)
        ...

    def run(self):
        for epoch in self.epochs:
            self.train()
            self.eval()
```
In this example we use the `DefaultExperiment` class and create a new experiment. 
The `@config` decorator injects the `defaults.config.json` into the `self.config` instance in a sub-property `experiments` of the default config.
One can then access the properties by de-referencing `self.config.experiments.<property>`.
`self.model` and `self.metric` are injected by the base class and are usually referenced in the `modules/defaults.config.json`.

### Decorators

It is also possible to simply use the required decorators without base class sub-classing. 
There are currently five decorators:
* `@config` - injects the default config instance and allows to load new configs
* `@context` - injects the application context which holds config and registry instances
* `@transform` - injects the default data transform instance
* `@dataset` - injects the default dataset instance
* `@model` - injects the default model instance
* `metric` - injects the default metric instance
* `@reference` - injects a singe defined instance
* `@references` - injects a set of pre-set or defined instance(s)
* `@strategy` - loads and defines a training set strategy from a config and injects the main object instances
* `@register` - allows to quickly register a new type to the context registry and inject the instance into the current instance
* `@hook` - allows to hook and overwrite an existing method to change the execution logic
* `@inject` - allows to inject single instances into different context options (registry, types, instances, configs)

By default the templates for a new project are creating a config file at the `configs/modules.config.json` path, which uses the following names for the dependency injection `['dataset', 'data_builder', 'criterion', 'model', 'writer', 'optimizer', 'transfom', 'metric', 'collectible']`.
An experiment gets these instances automatically injected.
```json
{
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
```

But it is simple to inject custom objects or exchangeable names by referring to a new name or type:
```json
{
  ...
  "other_model_name": "type::models.other.Network",
  ...
}
```

This now allows to load a model using the `@model` decorator withe the new property reference.

```python
class Demo:
    @references(properties=['other_model_name'])
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

