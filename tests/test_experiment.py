import os
import unittest

from lighter.config import Config
from lighter.context import Context
from examples.coco_looking.experiments.defaults import SimpleExperiment
from lighter.decorator import inject, config, device
from lighter.misc import create_template_file

DEFAULT_CONFIG_FILE = 'examples/coco_looking/configs/coco_looking.config.json'


class TestExperiment(unittest.TestCase):
    def test_config(self):
        Context.create(DEFAULT_CONFIG_FILE)
        config = Config.get_instance()
        # test if recursive import works
        self.assertTrue(config.modules.model is not None and not isinstance(config.modules.model, str))

    def test_decorator(self):
        Context.create(DEFAULT_CONFIG_FILE)
        exp = SimpleExperiment()
        self.assertTrue(exp.model is not None)

    def test_experiment_config_injection(self):
        exp = SimpleExperiment()
        self.assertTrue(exp.config.experiment.epochs == 50)

    def test_package_resource_access(self):
        project_name = 'tmp'
        module = 'models'
        template = {'project': project_name}
        create_template_file(project_name, module, 'defaults.template', 'defaults.py', template)
        self.assertTrue(os.path.exists('tmp/models/defaults.py'))

    def test_inject_decorator(self):
        class Demo:
            @device(id='cuda:0')
            @config(path='tests/test_inject_decorator.json', group='modules')
            @inject(instance='test', name='demo_model')
            def __init__(self):
                pass
        Context.create()
        demo = Demo()
        self.assertTrue(demo.demo_model is not None)
        self.assertTrue(demo.device is not None)


if __name__ == '__main__':
    unittest.main()
