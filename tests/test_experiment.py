import unittest

from lighter.config import Config
from lighter.context import Context
from examples.coco_looking.experiments.defaults import SimpleExperiment


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

    def test_execution_loop(self):
        exp = SimpleExperiment()
        exp()
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
