from pyspark.sql import SparkSession
import os

spark = SparkSession.builder.appName("FeatureReplaceMissing").getOrCreate()

def run(input_data_path, output_data_path):
    print("Replacing missing values with Spark...")

    df = spark.read.option("header", "true").csv(input_data_path)

    numeric_cols = [f.name for f in df.schema.fields if str(f.dataType) in ['IntegerType', 'DoubleType', 'LongType', 'FloatType']]

    for col in numeric_cols:
        median_val = df.approxQuantile(col, [0.5], 0.01)[0]
        df = df.na.fill({col: median_val})

    df.write.mode("overwrite").option("header", True).csv(output_data_path)
    print(f"Missing values replaced. Output saved to {output_data_path}")
