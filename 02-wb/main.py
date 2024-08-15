import modal
from infra import APP_NAME, DATA_VOLUME_CONFIG, MINUTES, wb_image

app = modal.App(
    name=APP_NAME,
    image=wb_image,
    volumes=DATA_VOLUME_CONFIG,
)

with app.image.imports():
    import os
    import pickle

    import requests
    import wandb
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, mean_squared_error

@app.function(
    secrets=[modal.Secret.from_name("my-wandb-secret")],
)
def intro():
    wandb.init(project="mlops-zoomcamp-wandb", name="experiment-1")

    X, y = load_iris(return_X_y=True)
    label_names = ["Setosa", "Versicolour", "Virginica"]

    # Log your model configs to Weights & Biases
    params = {"C": 0.1, "random_state": 42}
    wandb.config = params

    model = LogisticRegression(**params).fit(X, y)
    y_pred = model.predict(X)
    y_probas = model.predict_proba(X)

    wandb.log({
        "accuracy": accuracy_score(y, y_pred),
        "mean_squared_error": mean_squared_error(y, y_pred)
    })

    wandb.sklearn.plot_roc(y, y_probas, labels=label_names)

    wandb.sklearn.plot_precision_recall(y, y_probas, labels=label_names)

    wandb.sklearn.plot_confusion_matrix(y, y_pred, labels=label_names)

    # Save your model
    with open("logistic_regression.pkl", "wb") as f:
        pickle.dump(model, f)

# Log your model as a versioned file to Weights & Biases Artifact
    artifact = wandb.Artifact(f"iris-logistic-regression-model", type="model")
    artifact.add_file("logistic_regression.pkl")
    wandb.log_artifact(artifact)

    wandb.finish()

@app.function(
    timeout=4 * MINUTES,
)
def download_data():
    # URLs for the Titanic dataset
    train_url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    test_url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic_test.csv'

    # File paths
    train_path = os.path.join('/data', 'train.csv')
    test_path = os.path.join('/data', 'test.csv')

    # Download and save the train.csv file
    response = requests.get(train_url)
    with open(train_path, 'wb') as file:
        file.write(response.content)

    # Download and save the test.csv file
    response = requests.get(test_url)
    with open(test_path, 'wb') as file:
        file.write(response.content)

    print("Downloaded train.csv and test.csv to the data directory.")

@app.function(
    secrets=[modal.Secret.from_name("my-wandb-secret")],
)
def loading_data():
    import wandb

    # Initialize a WandB Run
    wandb.init(project="mlops-zoomcamp-wandb", job_type="log_data")

    # Log the `data` directory as an artifact
    artifact = wandb.Artifact('Titanic', type='dataset', metadata={"Source": "https://www.kaggle.com/competitions/titanic/data"})
    artifact.add_dir('/data')
    wandb.log_artifact(artifact)

    # End the WandB Run
    wandb.finish()

@app.function(
    secrets=[modal.Secret.from_name("my-wandb-secret")],
)
def version_data():
    import pandas as pd
    import wandb

    # Initialize a WandB Run
    wandb.init(project="mlops-zoomcamp-wandb", job_type="log_data")

    # Fetch the dataset artifact 
    artifact = wandb.use_artifact('andy4988/mlops-zoomcamp-wandb/Titanic:v0', type='dataset')
    artifact_dir = artifact.download()

    train_df = pd.read_csv(os.path.join(artifact_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(artifact_dir, "test.csv"))

    num_train_examples = int(0.8 * len(train_df))
    num_val_examples = len(train_df) - num_train_examples

    train_df["Split"] = ["Train"] * num_train_examples + ["Validation"] * num_val_examples
    train_df.to_csv("/data/train.csv", encoding='utf-8', index=False)

    # Log the `data` directory as an artifact
    artifact = wandb.Artifact('Titanic', type='dataset', metadata={"Source": "https://www.kaggle.com/competitions/titanic/data"})
    artifact.add_dir('/data')
    wandb.log_artifact(artifact)

    # End the WandB Run
    wandb.finish()

@app.function(
    secrets=[modal.Secret.from_name("my-wandb-secret")],
)
def explore_data():
    import pandas as pd
    import wandb

    # Initialize a WandB Run
    wandb.init(project="mlops-zoomcamp-wandb", job_type="explore_data")

    # Fetch the latest version of the dataset artifact 
    artifact = wandb.use_artifact('andy4988/mlops-zoomcamp-wandb/Titanic:latest', type='dataset')
    artifact_dir = artifact.download()

    # Read the files
    train_val_df = pd.read_csv(os.path.join(artifact_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(artifact_dir, "test.csv"))

    # Create tables corresponding to datasets
    train_val_table = wandb.Table(dataframe=train_val_df)
    test_table = wandb.Table(dataframe=test_df)

    # Log the tables to Weights & Biases
    wandb.log({
        "Train-Val-Table": train_val_table,
        "Test-Table": test_table
    })

    # End the WandB Run
    wandb.finish()

@app.function(
    secrets=[modal.Secret.from_name("my-wandb-secret")],
)
def base_line():
    import pandas as pd
    import wandb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score)

    # Initialize a WandB Run
    wandb.init(project="mlops-zoomcamp-wandb", name="baseline_experiment-2", job_type="train")

    # Fetch the latest version of the dataset artifact 
    artifact = wandb.use_artifact('andy4988/mlops-zoomcamp-wandb/Titanic:latest', type='dataset')
    artifact_dir = artifact.download()

    # Read the files
    train_val_df = pd.read_csv(os.path.join(artifact_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(artifact_dir, "test.csv"))

    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X_train = pd.get_dummies(train_val_df[features][train_val_df["Split"] == "Train"])
    X_val = pd.get_dummies(train_val_df[features][train_val_df["Split"] == "Validation"])
    y_train = train_val_df["Survived"][train_val_df["Split"] == "Train"]
    y_val = train_val_df["Survived"][train_val_df["Split"] == "Validation"]
    model_params = {"n_estimators": 100, "max_depth": 10, "random_state": 1}
    wandb.config = model_params

    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_probas_train = model.predict_proba(X_train)
    y_pred_val = model.predict(X_val)
    y_probas_val = model.predict_proba(X_val)

    wandb.log({
        "Train/Accuracy": accuracy_score(y_train, y_pred_train),
        "Validation/Accuracy": accuracy_score(y_val, y_pred_val),
        "Train/Presicion": precision_score(y_train, y_pred_train),
        "Validation/Presicion": precision_score(y_val, y_pred_val),
        "Train/Recall": recall_score(y_train, y_pred_train),
        "Validation/Recall": recall_score(y_val, y_pred_val),
        "Train/F1-Score": f1_score(y_train, y_pred_train),
        "Validation/F1-Score": f1_score(y_val, y_pred_val),
    })

    label_names = ["Not-Survived", "Survived"]

    wandb.sklearn.plot_class_proportions(y_train, y_val, label_names)
    wandb.sklearn.plot_summary_metrics(model, X_train, y_train, X_val, y_val)
    wandb.sklearn.plot_roc(y_val, y_probas_val, labels=label_names)
    wandb.sklearn.plot_precision_recall(y_val, y_probas_val, labels=label_names)
    wandb.sklearn.plot_confusion_matrix(y_val, y_pred_val, labels=label_names)

    # Save your model
    with open("random_forest_classifier.pkl", "wb") as f:
        pickle.dump(model, f)

    # Log your model as a versioned file to Weights & Biases Artifact
    artifact = wandb.Artifact(f"titanic-random-forest-model", type="model")
    artifact.add_file("random_forest_classifier.pkl")
    wandb.log_artifact(artifact)

    # End the WandB Run
    wandb.finish()


@app.function(
    secrets=[modal.Secret.from_name("my-wandb-secret")],
)
def hyper_op():
    import os
    import pickle

    import pandas as pd
    import wandb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score)
    
    def run_train():
        # Initialize a WandB Run
        wandb.init(project="mlops-zoomcamp-wandb", name="hyper_op_experiment", job_type="train")

        # Get hyperparameters from the run configs
        config = wandb.config

        # Fetch the latest version of the dataset artifact 
        artifact = wandb.use_artifact('andy4988/mlops-zoomcamp-wandb/Titanic:latest', type='dataset')
        artifact_dir = artifact.download()

        # Read the files
        train_val_df = pd.read_csv(os.path.join(artifact_dir, "train.csv"))

        features = ["Pclass", "Sex", "SibSp", "Parch"]
        X_train = pd.get_dummies(train_val_df[features][train_val_df["Split"] == "Train"])
        X_val = pd.get_dummies(train_val_df[features][train_val_df["Split"] == "Validation"])
        y_train = train_val_df["Survived"][train_val_df["Split"] == "Train"]
        y_val = train_val_df["Survived"][train_val_df["Split"] == "Validation"]

        # Define and Train RandomForestClassifier model
        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            bootstrap=config.bootstrap,
            warm_start=config.warm_start,
            class_weight=config.class_weight,
        )
        model.fit(X_train, y_train)

        # Make Predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_probas_val = model.predict_proba(X_val)

        # Log Metrics to Weights & Biases
        wandb.log({
            "Train/Accuracy": accuracy_score(y_train, y_pred_train),
            "Validation/Accuracy": accuracy_score(y_val, y_pred_val),
            "Train/Presicion": precision_score(y_train, y_pred_train),
            "Validation/Presicion": precision_score(y_val, y_pred_val),
            "Train/Recall": recall_score(y_train, y_pred_train),
            "Validation/Recall": recall_score(y_val, y_pred_val),
            "Train/F1-Score": f1_score(y_train, y_pred_train),
            "Validation/F1-Score": f1_score(y_val, y_pred_val),
        })

        # Plot plots to Weights & Biases
        label_names = ["Not-Survived", "Survived"]
        wandb.sklearn.plot_class_proportions(y_train, y_val, label_names)
        wandb.sklearn.plot_summary_metrics(model, X_train, y_train, X_val, y_val)
        wandb.sklearn.plot_roc(y_val, y_probas_val, labels=label_names)
        wandb.sklearn.plot_precision_recall(y_val, y_probas_val, labels=label_names)
        wandb.sklearn.plot_confusion_matrix(y_val, y_pred_val, labels=label_names)

        # Save your model
        with open("random_forest_classifier.pkl", "wb") as f:
            pickle.dump(model, f)

        # Log your model as a versioned file to Weights & Biases Artifact
        artifact = wandb.Artifact("titanic-random-forest-model", type="model")
        artifact.add_file("random_forest_classifier.pkl")
        wandb.log_artifact(artifact)
    
    SWEEP_CONFIG = {
        "method": "bayes",
        "metric": {"name": "Validation/Accuracy", "goal": "maximize"},
        "parameters": {
            "max_depth": {
                "distribution": "int_uniform",
                "min": 1,
                "max": 20,
            },
            "n_estimators": {
                "distribution": "int_uniform",
                "min": 10,
                "max": 100,
            },
            "min_samples_split": {
                "distribution": "int_uniform",
                "min": 2,
                "max": 10,
            },
            "min_samples_leaf": {
                "distribution": "int_uniform",
                "min": 1,
                "max": 4,
            },
            "bootstrap": {"values": [True, False]},
            "warm_start": {"values": [True, False]},
            "class_weight": {"values": ["balanced", "balanced_subsample"]},
        },
    }
    sweep_id = wandb.sweep(SWEEP_CONFIG, project="mlops-zoomcamp-wandb")
    wandb.agent(sweep_id, run_train, count=5)