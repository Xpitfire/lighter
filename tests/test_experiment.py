import os
import unittest

from lighter.config import Config
from lighter.context import Context
from examples.coco_looking.experiments.defaults import SimpleExperiment
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


if __name__ == '__main__':
    unittest.main()
