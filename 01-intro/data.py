from typing import Dict, List

import pandas as pd
from sklearn.feature_extraction import DictVectorizer


def load_transform_data(data_path):
    """
    Load and transform data from a Parquet file.

    Parameters:
    data_path (str): The path to the Parquet file.

    Returns:
    X, Y: The features and target values.
    """
    try:
        df = pd.read_parquet(data_path)
    except Exception as e:
        raise ValueError(f"Error reading the Parquet file: {e}")

    df['duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]

    categorical_columns = ['PULocationID', 'DOLocationID']
    numerical_columns = ['trip_distance']

    df[categorical_columns] = df[categorical_columns].astype(str)
    data_to_transform = df[categorical_columns + numerical_columns].to_dict(orient='records')
    dv = DictVectorizer()
    X = dv.fit_transform(data_to_transform)
    target = 'duration'
    Y = df[target].values

    return X, Y