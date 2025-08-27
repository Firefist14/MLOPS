from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
import os

spark = SparkSession.builder.appName("SplitData").getOrCreate()

def run(input_data_path, train_output_path, test_output_path, train_fraction=0.8):
    print("Splitting data with Spark...")

    df = spark.read.option("header", "true").csv(input_data_path)

    # Randomly split into train and test sets
    train_df, test_df = df.randomSplit([train_fraction, 1 - train_fraction], seed=42)

    os.makedirs(train_output_path, exist_ok=True)
    os.makedirs(test_output_path, exist_ok=True)

    train_df.write.mode("overwrite").option("header", True).csv(train_output_path)
    test_df.write.mode("overwrite").option("header", True).csv(test_output_path)

    print(f"Data split complete. Train saved to {train_output_path}, Test saved to {test_output_path}.")
