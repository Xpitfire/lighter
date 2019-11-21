import pickle

from lighter.misc import DotDict
from lighter.config import Config
from lighter.context import Context
from examples.coco_looking.datasets.defaults import LookingDataset
from box import Box


if __name__ == '__main__':
    test_json = {'test': {"test": "test"}}
    b = Box(test_json)
    parent, name = DotDict.resolve(b, 'test.test')
    print(parent, name)

    Context.create(config_dict={
        "strategy": {
            "model_import": "import::configs/sgd/alexnet.modules.config.json",
            "coco_import": "import::configs/sgd/coco_looking.config.json",
        }
    })
    test_str = pickle.dumps(test_json)

    class Demo(dict):
        pass
    test2_str = pickle.dumps(Demo(test_json))

    b = Box(test_json)
    print(b)
    assert 'test' == b.test.test
    testdot_str = pickle.dumps(b)

    config = Config.get_instance()
    config_str = pickle.dumps(config)
    dataset = LookingDataset()
    dataset_str = pickle.dumps(dataset)
    assert dataset_str is not None
