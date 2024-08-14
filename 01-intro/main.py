import modal
from data import load_transform_data
from infra import APP_NAME, MINUTES, VOLUME_CONFIG, app_image

app = modal.App(
    name=APP_NAME,
    image=app_image,
    volumes=VOLUME_CONFIG,
)

with app.image.imports():
    import pathlib
    import subprocess
    import time

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import root_mean_squared_error

@app.function(
    timeout=4 * MINUTES,
)
def download_data():
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    data_names = ["green_tripdata_2023-01.parquet", "green_tripdata_2023-02.parquet"]
    for name in data_names:
        dest_path = pathlib.Path("/data", name)
        print(f"Downloading {name} from {url}")
        command = f"wget {url}/{name} -O {dest_path}"
        subprocess.run(command, shell=True, check=True)
        print(f"Download of {name} completed successfully.")


@app.function()
def train():
    X_train, Y_train = load_transform_data("/data/green_tripdata_2023-01.parquet")
    lr = LinearRegression()
    lr.fit(X_train, Y_train)
    Y_train_pred = lr.predict(X_train)
    rmse = root_mean_squared_error(Y_train, Y_train_pred)
    return rmse

@app.local_entrypoint()
def main():
    t0 = time.time()
    rmse = train.remote()
    print(f"RMSE: {rmse}")
    print(f"Execution time: {time.time() - t0:.2f} seconds")