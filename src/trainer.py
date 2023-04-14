import argparse
from pandas import DataFrame, Series
import numpy as np
import pandas as pd
import pickle
import logging

from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .preprocess import preprocess, split_for_validation, CAT_FEATURES


class Trainer:
    """
    Class for training and predicting using sklearn models
    """

    def __init__(self, model, train_path: str, test_path: str):
        self.model = model

        self.train_df = preprocess(pd.read_csv(train_path, parse_dates=['datetime']), is_train=True)
        self.test_df = preprocess(pd.read_csv(test_path, parse_dates=['datetime']), is_train=False)

        self.is_trained = False

    def fit(self, with_validation=False):
        """
        Fits the model
        """

        logging.info("Training...")
        if with_validation:
            X_train, X_val, y_train, y_val = split_for_validation(self.train_df)
        else:
            X_train, y_train = self.train_df.drop('count', axis=1), self.train_df['count']
            X_val, y_val = None, None

        print(list(map(len, [X_train, y_train, X_val, y_val])))

        self.model.fit(X_train, y_train)
        self.is_trained = True

        return self.score(X_train, y_train, mean_squared_error), \
               (self.score(X_val, y_val, mean_squared_error) if with_validation else None)

    def predict(self, X: DataFrame) -> np.ndarray:
        """

        :param X: DataFrame
        :return: predictions from trained model
        """

        assert self.is_trained, "Model is not trained"

        return self.model.predict(X)

    def score(self, X: DataFrame, y: Series, score_function):
        """
        Scores model with given dataset and score_functoin
        :param X: DataFrame - features
        :param y: Series - ground truth
        :param score_function: function for scoring
        :return: float - calculated score
        """

        return score_function(y, self.predict(X))

    def save_model(self, path: str):
        """
        Saves trained model

        :param path: path to save the model
        :return: None
        """

        assert self.is_trained, "Model is not trained"

        pickle.dump(self.model, open(path, 'wb'))

    def get_train(self):
        """
        Returns train df
        :return: DataFrame
        """
        return self.train_df

    def get_test(self):
        """
        Returns test df
        :return: DataFrame
        """
        return self.test_df

    @staticmethod
    def default_trainer(train_path: str, test_path: str) -> 'Trainer':
        """

        :param train_path:
        :param test_path:
        :return: Trainer - trainer with RandomForestClassifier
        """
        model = Pipeline([
            ('onehot_encoder', make_column_transformer((OneHotEncoder(), CAT_FEATURES), remainder='passthrough')),
            ('regressor', RandomForestRegressor())
        ])
        return Trainer(model, train_path, test_path)

    @staticmethod
    def from_pretrained(model, train_path: str, test_path: str) -> 'Trainer':
        """

        :param train_path:
        :param test_path:
        :return: Trainer - trainer with pretrained model
        """
        trainer = Trainer(model, train_path, test_path)
        trainer.is_trained = True
        return trainer


def main():
    parser = argparse.ArgumentParser(prog='BikeSharingDemandRegression')
    parser.add_argument("--train", default="../data/train.csv")
    parser.add_argument("--test", default="../data/test.csv")
    parser.add_argument("--test_preds_out")
    parser.add_argument("--model_save_path")
    args = parser.parse_args()

    trainer = Trainer.default_trainer(args.train, args.test)
    train_mse, val_mse = trainer.fit(with_validation=True)

    logging.info(f"Train MSE {train_mse} | Val MSE {val_mse}")

    test_predictions = trainer.predict(trainer.get_test())

    pd.DataFrame({"test_preds": test_predictions}).to_csv(args.test_preds_out, header=True, index=False)

    trainer.save_model(args.model_save_path)


if __name__ == '__main__':
    main()
