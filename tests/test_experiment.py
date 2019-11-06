import unittest

from lighter.experiment import Experiment


class TestExperiment(unittest.TestCase):
    def test_execution_loop(self):
        exp = Experiment()
        self.assertTrue(exp())
