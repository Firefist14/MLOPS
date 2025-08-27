from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, dayofmonth, hour
from pyspark.sql.functions import col
import os

spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()

def preprocess_data(df):
    # Convert 'Issued_date' to timestamp
    df = df.withColumn("Issued_date", col("Issued_date").cast("timestamp"))

    # Extract datetime parts
    df = df.withColumn("Issued_year", year("Issued_date"))\
        .withColumn("Issued_month", month("Issued_date"))\
        .withColumn("Issued_day", dayofmonth("Issued_date"))\
        .withColumn("Issued_hour", hour("Issued_date"))\
        .drop("Issued_date")

    # Drop id columns if present
    for col_name in ['Unit_ID', 'Violation_ID', 'Tract']:
        if col_name in df.columns:
            df = df.drop(col_name)

    # Fill missing numeric values with median approximation
    numeric_cols = [f.name for f in df.schema.fields if str(f.dataType) in ['IntegerType', 'DoubleType', 'LongType', 'FloatType']]
    for ncol in numeric_cols:
        median_val = df.approxQuantile(ncol, [0.5], 0.01)[0]
        df = df.na.fill({ncol: median_val})

    # Encode categorical columns using StringIndexer
    from pyspark.ml.feature import StringIndexer
    cat_cols = [c for c, t in df.dtypes if t == 'string']
    for c in cat_cols:
        indexer = StringIndexer(inputCol=c, outputCol=c+"_index", handleInvalid='keep')
        df = indexer.fit(df).transform(df).drop(c).withColumnRenamed(c+"_index", c)

    return df

def run(input_data_path, output_data_path):
    print("Running feature engineering with Spark preprocessing...")

    if os.path.isfile(input_data_path):
        df = spark.read.option("header", "true").csv(input_data_path)
    else:
        df = spark.read.option("header", "true").csv(input_data_path)

    df = preprocess_data(df)

    df.write.mode("overwrite").option("header", True).csv(output_data_path)
    print(f"Feature engineering complete. Output saved to {output_data_path}")
