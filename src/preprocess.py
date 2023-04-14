from pandas import DataFrame, Series
import typing as tp

from sklearn.model_selection import train_test_split

CAT_FEATURES = ["weather", "season", "weekday"]


def preprocess(df: DataFrame, *, is_train: bool) -> DataFrame:
    """

    Function for data preprocessing: drops unused columns, generates new columns from datetime and casts categorical
    features to category

    :param df: DataFrame - dataframe for preprocessing
    :param is_train: bool - for train columns dropping
    :return: df: DataFrame - preprocessed DataFrame
    """
    if is_train:
        df = df.drop(['casual', 'registered'], axis=1)
    df['year'] = df.datetime.apply(lambda x: x.year)
    df['month'] = df.datetime.apply(lambda x: x.month)
    df['day'] = df.datetime.apply(lambda x: x.day)
    df['hour'] = df.datetime.apply(lambda x: x.hour)
    df['weekday'] = df.datetime.dt.day_name()

    df = df.drop('datetime', axis=1)

    for feature in CAT_FEATURES:
        df[feature] = df[feature].astype('category')
    return df


def split_for_validation(train_df: DataFrame, split_seed=42) -> tp.Tuple[DataFrame, Series, DataFrame, Series]:
    """
    Splits train datafram into train and validation dataframes

    :param train_df: DataFrame - train dataframe
    :param split_seed: int - seed for train_test_split
    :return: X_train, y_train, X_val, y_val - 4 dataframes for train and validation
    """
    X = train_df.drop(['count'], axis=1)
    y = train_df['count']
    return train_test_split(X, y, test_size=0.2, shuffle=True, random_state=split_seed)


__all__ = [
    "preprocess",
    "split_for_validation",
    "CAT_FEATURES"
]
