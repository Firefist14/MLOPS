from pyspark.sql import SparkSession
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

spark = SparkSession.builder.appName("TrainModel").getOrCreate()

def run(train_data_path, test_data_path, model_output_path):
    print("Training model with Spark and sklearn...")

    train_df = spark.read.option("header", "true").csv(train_data_path).toPandas()
    test_df = spark.read.option("header", "true").csv(test_data_path).toPandas()

    target_column = "PaymentIsOutstanding"
    le = LabelEncoder()

    train_df[target_column] = le.fit_transform(train_df[target_column].astype(str))
    test_df[target_column] = le.transform(test_df[target_column].astype(str))

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    os.makedirs(model_output_path, exist_ok=True)
    model_file = os.path.join(model_output_path, "model.joblib")
    joblib.dump(model, model_file)

    print(f"Model training complete. Model saved to {model_file}")
