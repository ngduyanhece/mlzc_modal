from pathlib import PurePosixPath
from typing import Union

import modal

APP_NAME = "02-mlzc-wb"

MINUTES = 60

wb_image = (
    modal.Image.debian_slim()
    .pip_install("pandas")
    .pip_install("matplotlib")
    .pip_install("scikit-learn")
    .pip_install("wandb")
    .pip_install("pyarrow")
)

# Volumes for pre-trained models and training runs.
data_volume = modal.Volume.from_name(
    "mlzc-data-vol", create_if_missing=True
)


DATA_VOLUME_CONFIG: dict[Union[str, PurePosixPath], modal.Volume] = {
    "/data": data_volume,
}