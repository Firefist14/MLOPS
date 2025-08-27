from pyspark.sql import SparkSession
import os

spark = SparkSession.builder.appName("FeatureSelection").getOrCreate()

def run(input_data_path, output_data_path):
    print("Selecting features with Spark...")

    df = spark.read.option("header", "true").csv(input_data_path)

    drop_cols = ['Tract', 'License_Plate_State', 'Unit_ID', 'Violation_ID']
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(c)

    df.write.mode("overwrite").option("header", True).csv(output_data_path)
    print(f"Feature selection complete. Output saved to {output_data_path}")
