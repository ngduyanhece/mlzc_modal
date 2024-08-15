import modal
from infra import MINUTES, VOLUME_CONFIG, mlflow_image

app = modal.App(
    name="mlflow",
    image=mlflow_image,
    volumes=VOLUME_CONFIG,
)

with app.image.imports():
    import pathlib
    import subprocess

@app.function(
    timeout=24 * 60 * MINUTES
)
@modal.web_server(8080, startup_timeout=10 * MINUTES)
def track():
    sqilte_db = pathlib.Path("/mlflow_db", "mydb.sqlite")
    cmd = f"mlflow server --backend-store-uri sqlite:///{sqilte_db} --host 0.0.0.0 --port 8080"
    subprocess.Popen(cmd, shell=True)