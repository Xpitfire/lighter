# Lighter

Lightweight extension for torch to speed up project prototyping and enable dependency injection of object instances.
This framework includes a setup script to get started with a pre-defined project structure and the documentation offers examples and an overview of the initial usage.

## Install

```shell script
$> pip install torch-lighter
```

## Description

Create new project structure:
```shell script
$> lighter-init <project-name>
``` 

### Structure overview
The following project structure will be created:
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
- experiments: experiment logic files containing the main `train()` and `eval()` method
- metrics: metrics used to measure performance
- modules: references all available modules
- optimizers: optimizers used for performing the step
- writers: tensorboard writers

### Detailed description

Each directory containing a python file may also contain a config file outsourcing properties for testing.

## Usage

### Inheritance from base class

One can simply extend from any base class within the lighter framework:
```python
class SimpleExperiment(BaseExperiment):
    @config(path='my_project/experiments/defaults.config.json', group='experiments')
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
The `@config` decorator injects the `defaults.config.json` into the `self.config` instance in the `experiments` group.
One can then access the properties by de-referencing `self.config.experiments.<property>`.
`self.model` and `self.metric` are injected by the base class and are usually referenced in the `modules/defaults.config.json`.

### Decorators

It is also possible to simply use the required decorators without base class sub-classing. 
There are currently five decorators:
* `@config` - injects the default config instance and allows to load new configs
* `@context` - injects the application context which holds config and registry instances
* `@dataset` - injects dataset instance
* `@model` - injects model instance
* `@experiment` - injects a set of pre-set instances

By default the templates for a new project are using these names for dependency injection `['dataset', 'data_builder', 'criterion', 'model', 'writer', 'optimizer', 'metric', 'collectible']`
```json
{
    "modules": {
        "dataset": "type::$project.datasets.defaults.LookingDataset",
        "data_builder": "type::$project.data_builders.defaults.SimpleDataBuilder",
        "criterion": "type::$project.criterions.defaults.SimpleCriterion",
        "model": "type::$project.models.defaults.AlexNetFeatureExtractionNetwork",
        "writer": "type::$project.writers.defaults.SimpleWriter",
        "optimizer": "type::$project.optimizers.defaults.SimpleOptimizer",
        "metric": "type::$project.metrics.defaults.SimpleMetric",
        "collectible": "type::$project.collectibles.defaults.SimpleCollectible"
    }
}
```

But it is simply exchangeable by referring to a new name such as:
```json
{
    "modules": {
        ...
        "other_model_name": "type::$project.models.defaults.AlexNetFeatureExtractionNetwork",
        ...
    }
}
```


```python
class Demo:
    @model(name='other_model_name')
    def __init__(self):
        ...
        self.other_model_name.<property>
        ...
```

### Config templates

All configs are based on `json` files. Yet, when importing the configs `lighter` parses some escape sequences:
* `config::<config-file-path>` - imports another config into the current json config
* `import::<config-file-path>` - imports all properties of the referenced file into the current config instance (attention may override existing properties)
* `type::<type-file-path>` - imports a python type which is registered to the context types registry and can be used to instantiate new objects
