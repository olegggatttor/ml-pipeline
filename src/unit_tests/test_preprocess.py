import unittest
import configparser
from pandas import DataFrame
from ..trainer import Trainer

config = configparser.ConfigParser()
config.read("config.ini")


class TestTraining(unittest.TestCase):

    def setUp(self) -> None:
        self.trainer = Trainer.default_trainer(config['unit_testing']['train_path'],
                                               config['unit_testing']['test_path'])

    def test_get_train_data(self):
        self.assertTrue(type(self.trainer.get_train()) is DataFrame)

    def test_get_test_data(self):
        self.assertTrue(type(self.trainer.get_test()) is DataFrame)

    def test_train_columns_preprocessed(self):
        self.assertTrue('datetime' not in self.trainer.get_train().columns)
        self.assertTrue('year' in self.trainer.get_train().columns)
        self.assertTrue('month' in self.trainer.get_train().columns)
        self.assertTrue('day' in self.trainer.get_train().columns)
        self.assertTrue('hour' in self.trainer.get_train().columns)

    def test_test_columns_preprocessed(self):
        self.assertTrue('datetime' not in self.trainer.get_test().columns)
        self.assertTrue('year' in self.trainer.get_test().columns)
        self.assertTrue('month' in self.trainer.get_test().columns)
        self.assertTrue('day' in self.trainer.get_test().columns)
        self.assertTrue('hour' in self.trainer.get_test().columns)


if __name__ == "__main__":
    unittest.main()
