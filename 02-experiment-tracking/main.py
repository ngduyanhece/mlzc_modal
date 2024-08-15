import modal
from data import load_and_transform_data
from infra import (APP_NAME, APP_TRACKING_URI, DATA_VOLUME_CONFIG,
                   EXPERIMENT_NAME, MINUTES, app_image, model_volume)

app = modal.App(
    name=APP_NAME,
    image=app_image,
    volumes=DATA_VOLUME_CONFIG,
)

with app.image.imports():
    import pathlib
    import pickle
    import subprocess
    import time

    import mlflow
    from mlflow.tracking import MlflowClient
    from sklearn.linear_model import Lasso
    from sklearn.metrics import mean_squared_error


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
def lr_exp():
    mlflow.set_tracking_uri(APP_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    X_train, Y_train, X_val, Y_val = load_and_transform_data(
        "/data/green_tripdata_2023-01.parquet",
        "/data/green_tripdata_2023-02.parquet"
    )
    with mlflow.start_run():

        mlflow.set_tag("developer", "andy")

        mlflow.log_param("train-data-path", "/data/green_tripdata_2023-01.parquet")
        mlflow.log_param("valid-data-path", "/data/green_tripdata_2023-02.parquet")

        alpha = 0.1
        mlflow.log_param("alpha", alpha)
        lr = Lasso(alpha)
        lr.fit(X_train, Y_train)

        Y_pred = lr.predict(X_val)
        rmse = mean_squared_error(Y_val, Y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        with open('lin_reg.bin"', 'wb') as f_out:
            pickle.dump(lr, f_out)
        mlflow.log_artifact(local_path='lin_reg.bin"', artifact_path="mlflow_db")
    return "Training completed successfully."

@app.function(
    timeout=30 * MINUTES,
)
def xg_boot_hp():
    import xgboost as xgb
    from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
    from hyperopt.pyll import scope

    mlflow.set_tracking_uri(APP_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    X_train, Y_train, X_val, Y_val = load_and_transform_data(
        "/data/green_tripdata_2023-01.parquet",
        "/data/green_tripdata_2023-02.parquet"
    )
    train = xgb.DMatrix(X_train, label=Y_train)
    valid = xgb.DMatrix(X_val, label=Y_val)
    
    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost", "andy")
            mlflow.log_params(params)
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=1000,
                evals=[(valid, 'validation')],
                early_stopping_rounds=10
            )
            Y_pred = booster.predict(valid)
            rmse = mean_squared_error(Y_val, Y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}
    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:linear',
        'seed': 42
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials()
    )


@app.function(
    timeout=30 * MINUTES,
)
def ensemble():
    from sklearn.ensemble import (ExtraTreesRegressor,
                                  GradientBoostingRegressor,
                                  RandomForestRegressor)
    from sklearn.svm import LinearSVR

    mlflow.set_tracking_uri(APP_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog()
    from mlflow.models import infer_signature

    # create models directory ìf it does not exist

    X_train, Y_train, X_val, Y_val = load_and_transform_data(
        "/data/green_tripdata_2023-01.parquet",
        "/data/green_tripdata_2023-02.parquet"
    )
    signature = infer_signature(X_train, Y_train)
    for model_class in (RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, LinearSVR):
    
        with mlflow.start_run():

            mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.parquet")
            mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.parquet")

            ml_model = model_class()
            ml_model.fit(X_train, Y_train)

            Y_pred = ml_model.predict(X_val)
            rmse = mean_squared_error(Y_val, Y_pred, squared=False)
            # pickle model
            with open(f'{model_class.__name__}.bin', 'wb') as f_out:
                pickle.dump(ml_model, f_out)
            mlflow.log_artifact(local_path=f'{model_class.__name__}.bin', artifact_path="mlflow_db")
            mlflow.log_metric("rmse", rmse)
            mlflow.sklearn.log_model(
                sk_model=ml_model,
                artifact_path="models",
                signature=signature,
                registered_model_name=f'nyc-taxi-{model_class.__name__}'
            )

@app.function(
    timeout=30 * MINUTES,
)
def create_registry_model():
    from mlflow.entities import ViewType
    mlflow.set_tracking_uri(APP_TRACKING_URI)

@app.function(
    timeout=30 * MINUTES,
)
def update_registry_model():
    client = MlflowClient(tracking_uri=APP_TRACKING_URI)
    model_name = "nyc-taxi-regressor-ensemble"
    run_id = "72f7587d5bc14c72b9c0c59792cbb89a"
    model_uri = 'runs:/72f7587d5bc14c72b9c0c59792cbb89a/model'
    result = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
        tags={"version": "1.0.0"},
        description="NYC Taxi regressor ensemble model"
    )
    return result.name

@app.function(
    timeout=30 * MINUTES,
)
def get_registry_model():
    client = MlflowClient(tracking_uri=APP_TRACKING_URI)
    model_name = "nyc-taxi-regressor-ensemble"
    latest_versions = client.get_latest_versions(name=model_name)

    for version in latest_versions:
        print(f"version: {version.version}")

@app.function(
    timeout=30 * MINUTES,
)
def eval_model():
    mlflow.set_tracking_uri(APP_TRACKING_URI)
    X_train, Y_train, X_val, Y_val = load_and_transform_data(
        "/data/green_tripdata_2023-01.parquet",
        "/data/green_tripdata_2023-02.parquet"
    )

    def test_model(model_name, model_version, X_test, Y_test):
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
        Y_pred = model.predict(X_test)
        return {"rmse": mean_squared_error(Y_test, Y_pred, squared=False)}

    model_name = "nyc-taxi-ExtraTreesRegressor"
    model_version = "2"
    result = test_model(model_name, model_version, X_val, Y_val)
    print(result)


# @app.function()
# def run():
#     with open("/models/xyz.txt", "w") as f:
#         f.write("hello")
#     model_volume.commit() 

@app.local_entrypoint()
def main():
    t0 = time.time()
    print(lr_exp.remote())
    print(f"Execution time: {time.time() - t0:.2f} seconds")
