import os
import unittest

from lighter.config import Config
from lighter.context import Context
from lighter.decorator import inject, config, device, context, search, strategy, InjectOption, reference, hook, \
    references
from lighter.experiment import DefaultExperiment
from lighter.utils.io import create_template_file, get_lighter_path
from lighter.parameter import GridParameter
from examples.coco_looking.experiments.defaults import Experiment

DEFAULT_CONFIG_FILE = 'configs/coco_looking.config.json'


class TestLighter(unittest.TestCase):
    def test_config(self):
        Context.create(DEFAULT_CONFIG_FILE)
        config = Config.get_instance()
        # test if recursive import works
        self.assertTrue(config.model is not None and not isinstance(config.model, str))

    def test_decorator(self):
        Context.create()
        exp = Experiment()
        self.assertTrue(exp.model is not None)

    def test_package_resource_access(self):
        Context.create()
        project_name = 'tmp'
        module = 'models'
        template = {'project': project_name}
        create_template_file(project_name, 'defaults.template', 'defaults.py', template, module_name=module)
        self.assertTrue(os.path.exists('tmp/models/defaults.py'))

    def test_inject_decorator(self):
        Context.create()

        class Demo:
            @device(name='cpu')
            @config(path='tests/test_inject_decorator.json', property='modules')
            @inject(source='test', property='demo_model', option=InjectOption.Instance)
            def __init__(self):
                # trick to help the auto-completion interpret the type at design-time
                self.demo_model = self.demo_model  # type: models.alexnet.AlexNetFeatureExtractionModel

        demo = Demo()
        self.assertTrue(demo.config.modules is not None)
        self.assertTrue(demo.demo_model is not None)
        self.assertTrue(demo.device is not None)

    def test_search_iterator(self):
        Context.create()
        assert_true = self.assertTrue

        class SearchExperiment:
            @context
            @search(params=[('demo', GridParameter(ref='test', min=1, max=10))])
            def __init__(self):
                self.config.set_value('test', 2)

            def run(self):
                self.search.demo.update_config(self.config)
                assert_true(self.search.demo.value == 2)
                for i, config in enumerate(self.search.demo):
                    if i == 0:
                        assert_true(config.test == 1)
                assert_true(self.search.demo.value == 11)
        exp = SearchExperiment()
        exp.run()

    def test_strategy(self):
        Context.create()

        class Demo:
            @strategy(config='configs/coco_looking.config.json')
            def __init__(self):
                pass
        exp = Demo()
        self.assertTrue(exp.model is not None)

    def test_lighter_init_path(self):
        exists, path = get_lighter_path('models', 'defaults.template')
        self.assertTrue(exists)

    def test_reference(self):
        Context.create(DEFAULT_CONFIG_FILE)

        class Demo:
            @reference(name='model')
            def __init__(self):
                pass
        demo = Demo()
        self.assertTrue(demo.model is not None)

    def test_hook(self):
        Context.create(DEFAULT_CONFIG_FILE)

        class State:
            def __init__(self):
                self.success = False
        s = State()

        def alternative_run(ori, args):
            args[0].success = True
            print("replace method")

        class Exp(DefaultExperiment):
            @reference(name='model')
            @hook(method='run', replace_with=alternative_run, args=[s])
            def __init__(self):
                super(Exp, self).__init__()

        exp = Exp()
        exp()
        self.assertTrue(s.success)

    def test_references(self):
        Context.create(DEFAULT_CONFIG_FILE)

        class Demo:
            def __init__(self):
                pass
        demo = Demo()
        Context.get_instance().registry.instances['demo'] = demo

        class Usage:
            @references(names=['demo'])
            def __init__(self):
                pass
        usage = Usage()
        self.assertTrue(usage.demo is not None)


if __name__ == '__main__':
    unittest.main()
