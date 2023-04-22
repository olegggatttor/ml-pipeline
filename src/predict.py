import argparse
import pickle
import numpy as np
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(prog='BikeSharingDemandRegression')
    parser.add_argument("--data", default="tests/func_samples.csv")
    parser.add_argument("--from_pretrained", default="data/r_forest.pickle")
    args = parser.parse_args()

    with open(args.from_pretrained, 'rb') as f:
        trainer = Trainer.from_pretrained(pickle.load(f), args.data, args.data)
        test_predictions = trainer.predict(trainer.get_test())

        assert np.allclose(test_predictions, trainer.get_test()['count'], rtol=0, atol=3)


if __name__ == '__main__':
    main()