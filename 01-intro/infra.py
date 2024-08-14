from pathlib import PurePosixPath
from typing import Union

import modal

APP_NAME = "01-mlzc-intro"
MINUTES = 60
HOURS = 60 * MINUTES

app_image = (
    modal.Image.debian_slim()
    .apt_install("wget")
    .pip_install("scikit-learn")
    .pip_install("seaborn")
    .pip_install("matplotlib")
    .pip_install("pyarrow")  # Add pyarrow
    .pip_install("fastparquet")  # Add fastparquet
)
# run_commands(
#     "wget https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-01.parquet",
#     "wget https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-02.parquet"
# )

# Volumes for pre-trained models and training runs.
data_volume = modal.Volume.from_name(
    "mlzc-data-vol", create_if_missing=True
)
VOLUME_CONFIG: dict[Union[str, PurePosixPath], modal.Volume] = {
    "/data": data_volume,
}