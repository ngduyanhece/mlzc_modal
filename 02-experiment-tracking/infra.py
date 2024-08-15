from pathlib import PurePosixPath
from typing import Union

import modal

APP_NAME = "02-mlzc-experiment-tracking"
APP_TRACKING_URI = "https://ngduyanhece--mlflow-track.modal.run"
EXPERIMENT_NAME = "nyc-taxi-experiment-ensemble"
MINUTES = 60
HOURS = 60 * MINUTES

mlflow_image = (
    modal.Image.debian_slim()
    .pip_install("mlflow")
)

# Volumes for mlflow tracking server.
mlflow_db_volume = modal.Volume.from_name(
    "mlflow-db-vol", create_if_missing=True
)
VOLUME_CONFIG: dict[Union[str, PurePosixPath], modal.Volume] = {
    "/mlflow_db": mlflow_db_volume,
}


# Volumes for pre-trained models and training runs.
data_volume = modal.Volume.from_name(
    "mlzc-data-vol", create_if_missing=True
)
model_volume = modal.Volume.from_name(
    "mlzc-model-vol", create_if_missing=True
)
DATA_VOLUME_CONFIG: dict[Union[str, PurePosixPath], modal.Volume] = {
    "/data": data_volume,
    "/models": model_volume,
}

app_image = (
    modal.Image.debian_slim()
    .apt_install("wget")
    .pip_install("scikit-learn")
    .pip_install("seaborn")
    .pip_install("matplotlib")
    .pip_install("pyarrow")  # Add pyarrow
    .pip_install("fastparquet")
    .pip_install("mlflow")
    .pip_install("xgboost")
    .pip_install("hyperopt")
)
