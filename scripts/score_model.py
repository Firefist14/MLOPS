import pandas as pd
import mlflow.sklearn
import os

# Load model once (could be done inside a function or notebook cell)
model_uri = "runs:/<your-run-id>/model"  # or local path in DBFS if needed
model = mlflow.sklearn.load_model(model_uri)

def run(mini_batch_files):
    print(f"Running scoring on {len(mini_batch_files)} files")
    data_frames = [pd.read_csv(f) for f in mini_batch_files]
    data = pd.concat(data_frames)
    predictions = model.predict(data)
    data['PaymentIsOutstanding'] = predictions
    return data
