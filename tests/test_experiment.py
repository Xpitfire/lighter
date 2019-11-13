import os
import unittest

from lighter.config import Config
from lighter.context import Context
from examples.coco_looking.experiments.defaults import SimpleExperiment
from lighter.decorator import inject, config, device, context, search, strategy, InjectOption
from lighter.misc import create_template_file
from lighter.parameter import GridParameter
from lighter.misc import get_lighter_path

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
        create_template_file(project_name, 'defaults.template', 'defaults.py', template, module_name=module)
        self.assertTrue(os.path.exists('tmp/models/defaults.py'))

    def test_inject_decorator(self):
        class Demo:
            @device(name='cuda:0')
            @config(path='tests/test_inject_decorator.json', property='modules')
            @inject(source='test', property='demo_model', option=InjectOption.Instance)
            def __init__(self):
                pass
        Context.create()
        demo = Demo()
        self.assertTrue(demo.config.modules is not None)
        self.assertTrue(demo.demo_model is not None)
        self.assertTrue(demo.device is not None)

    def test_search_iterator(self):
        assert_true = self.assertTrue

        class SearchExperiment:
            @context
            @search(params=[('demo', GridParameter(ref='test', min=1, max=10))])
            def __init__(self):
                self.config.set_value('test', 2)

            def run(self):
                assert_true(self.search.demo.value == 2)
                for i, param in enumerate(self.search.demo):
                    if i == 0:
                        assert_true(param == 1)
                assert_true(self.search.demo.value == 10)
        exp = SearchExperiment()
        exp.run()

    def test_strategy(self):
        class Experiment:
            @strategy(config='examples/coco_looking/configs/modules.config.json')
            def __init__(self):
                pass
        exp = Experiment()
        self.assertTrue(exp.model is not None)

    def test_lighter_init_path(self):
        exists, path = get_lighter_path('models', 'defaults.template')
        self.assertTrue(exists)


if __name__ == '__main__':
    unittest.main()
