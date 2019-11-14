import unittest

from lighter.decorator import register


class TestMineRL(unittest.TestCase):

    def test_client(self):
        class Example:
            @register(type='envs.defaults.Environment', property='env')
            def __init__(self):
                pass
        e = Example()
        self.assertTrue(hasattr(e.env, 'connect'))


if __name__ == '__main__':
    unittest.main()
