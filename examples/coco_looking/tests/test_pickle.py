import pickle

from lighter.misc import DotDict
from lighter.config import Config
from lighter.context import Context
from examples.coco_looking.datasets.defaults import LookingDataset
from box import Box


class DotDictTest(dict):
    """dot.notation access to dictionary attributes"""
    def get_value(self, name, default=None):
        if name in self:
            return self[name]
        else:
            return default

    def has_value(self, name):
        return name in self

    @staticmethod
    def resolve(parent_ori, name_ori):
        parent = parent_ori
        prev_parent = parent_ori
        groups = name_ori.split('.')
        for group in groups[:-1]:
            parent = parent.get_value(group)
            if parent is None:
                parent = DotDict()
                setattr(prev_parent, group, parent)
            else:
                prev_parent = parent
        return parent, groups[-1]


if __name__ == '__main__':
    Context.create('configs/sgd/alexnet.modules.config.json')
    test_json = {'test': {"test": "test"}}
    test_str = pickle.dumps(test_json)

    class Demo(dict):
        pass
    test2_str = pickle.dumps(Demo(test_json))
    test3_str = pickle.dumps(DotDictTest(test_json))

    b = Box(test_json)
    print(b)
    assert 'test' == b.test.test
    testdot_str = pickle.dumps(b)

    val = DotDict(test_json)
    print(val.__dict__)
    print(val)
    print(val['test'])
    test4_str = pickle.dumps(val)
    config = Config.get_instance()
    config_str = pickle.dumps(config)
    dataset = LookingDataset()
    dataset_str = pickle.dumps(dataset)
    assert dataset_str is not None
