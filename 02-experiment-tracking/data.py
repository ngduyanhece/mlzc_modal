from typing import Dict, List

import pandas as pd
from sklearn.feature_extraction import DictVectorizer


def load_and_transform_data(train_data_path, val_data_path):
    train_df = pd.read_parquet(train_data_path)
    # compute the duration of the trip
    
    train_df['duration'] = (train_df['lpep_dropoff_datetime'] - train_df['lpep_pickup_datetime']).dt.total_seconds() / 60
    # drop the outliers
    train_df = train_df[(train_df.duration >= 1) & (train_df.duration <= 60)].copy()
    categorical = ['PULocationID', 'DOLocationID']
    train_df[categorical] = train_df[categorical].astype(str)
    train_dicts = train_df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    target = 'duration'
    Y_train = train_df[target].values
    # validation data
    val_df = pd.read_parquet(val_data_path)
    val_df['duration'] = (val_df['lpep_dropoff_datetime'] - val_df['lpep_pickup_datetime']).dt.total_seconds() / 60
    val_df = val_df[(val_df.duration >= 1) & (val_df.duration <= 60)].copy()
    val_df[categorical] = val_df[categorical].astype(str)
    val_dicts = val_df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    Y_val = val_df[target].values
    return X_train, Y_train, X_val, Y_val