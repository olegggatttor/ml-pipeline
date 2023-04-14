import unittest
import configparser
from ..trainer import Trainer

config = configparser.ConfigParser()
config.read("config.ini")


class TestTraining(unittest.TestCase):

    def setUp(self) -> None:
        self.trainer = Trainer.default_trainer(config['unit_testing']['train_path'],
                                               config['unit_testing']['test_path'])

    def test_training(self):
        self.assertFalse(self.trainer.is_trained)
        train_mse, val_mse = self.trainer.fit(with_validation=True)
        self.assertTrue(train_mse is not None)
        self.assertTrue(train_mse is not None)
        self.assertTrue(self.trainer.is_trained)


if __name__ == "__main__":
    unittest.main()