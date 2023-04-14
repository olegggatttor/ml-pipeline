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

from preprocess import preprocess, split_for_validation, CAT_FEATURES


class Trainer:
    """
    Class for training and predicting using sklearn models
    """

    def __init__(self, model):
        self.model = model
        self.is_trained = False

    def fit(self, X: DataFrame, y: Series):
        """
        Fits the model
        :param X: DataFrame
        :param y: Series
        :return: None
        """

        logging.info("Training...")
        self.model.fit(X, y)
        self.is_trained = True

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='BikeSharingDemandRegression')
    parser.add_argument("--train", default="../data/train.csv")
    parser.add_argument("--test", default="../data/test.csv")
    parser.add_argument("--model_save_path")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train, parse_dates=['datetime'])
    test_df = pd.read_csv(args.test, parse_dates=['datetime'])

    train_df = preprocess(train_df, is_train=True)
    test_df = preprocess(test_df, is_train=False)

    X_train, y_train, X_val, y_val = split_for_validation(train_df)

    model = Pipeline([
        ('onehot_encoder', make_column_transformer((OneHotEncoder(), CAT_FEATURES), remainder='passthrough')),
        ('regressor', RandomForestRegressor())
    ])

    trainer = Trainer(model)
    trainer.fit(X_train, y_train)

    score_function = lambda y, pred: np.sqrt(mean_squared_error(y, pred))

    logging.info(f"Train RMSE: {trainer.score(X_train, y_train, score_function)}")
    logging.info(f"Val RMSE: {trainer.score(X_val, y_val, score_function)}")

    trainer.save_model(args.model_save_path)
