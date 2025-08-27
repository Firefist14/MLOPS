import mlflow
import mlflow.sklearn
import joblib
import os

def run(model_output_path):
    print("Registering model with MLflow...")

    model_file = os.path.join(model_output_path, "model.joblib")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file does not exist: {model_file}")

    model = joblib.load(model_file)

    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="my_model"  # Set your model registry name here
        )

    print("Model registered to MLflow.")
