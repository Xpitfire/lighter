import unittest

from lighter.experiment import SimpleExperiment


class TestExperiment(unittest.TestCase):
    def test_execution_loop(self):
        exp = SimpleExperiment()
        self.assertTrue(exp())
