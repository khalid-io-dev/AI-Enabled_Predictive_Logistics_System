# spark_batch_persist.py
"""
Batch script to load data, apply saved PipelineModel, persist predictions to MySQL,
compute aggregations and persist to MongoDB.

Updated to align with train_model.py which only outputs selected features:
['geo_distance_km', 'order_region', 'shipping_mode', 'customer_segment', 'sales', 'delivery_days', 'num_orders', 'late_delivery_risk']

Switched to MySQL for persistence.
"""

import os, traceback
import uuid
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.functions import vector_to_array
import pandas as pd

# Absolute paths
BASE_DIR = r"C:\Users\ADMIN\Desktop\BRIEF-7"
MODEL_PATH = os.path.join(BASE_DIR, "pipeline_model")
INPUT_PARQUET = os.path.join(BASE_DIR, "cleaned_batch_input.parquet")

# Config - MySQL
OUTPUT_PRED_TABLE = "predictions"
MYSQL_HOST = "127.0.0.1"
MYSQL_PORT = "3306"
MYSQL_USER = "root"
MYSQL_PASSWORD = ""
MYSQL_DB = "dataco"

# JDBC URL for MySQL
# Note: Spark needs the MySQL JDBC driver jar to be available in the classpath for .write.jdbc to work.
# If not available, it will fall back to the Python implementation.
MYSQL_JDBC_URL = f"jdbc:mysql://{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?useSSL=false"
MYSQL_PROPS = {"user": MYSQL_USER, "password": MYSQL_PASSWORD, "driver": "com.mysql.cj.jdbc.Driver"}

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.environ.get("MONGO_DB", "dataco_analytics")
MONGO_COLL = "batch_aggregates"

spark = (SparkSession.builder
         .appName("DataCo_Batch_Persist")
         .master("local[*]")
         .config("spark.sql.execution.arrow.pyspark.enabled", "false")
         # Attempt to load mysql driver if present, otherwise ignore
         # .config("spark.jars", "/path/to/mysql-connector-java.jar") 
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

print(f"Reading input from {INPUT_PARQUET}...")
try:
    df = spark.read.parquet(INPUT_PARQUET)
    print(f"Read {df.count()} rows.")
    print("Columns:", df.columns)
except Exception:
    print("Failed to read input parquet:", traceback.format_exc())
    spark.stop()
    raise

# load model
print(f"Loading model from {MODEL_PATH}...")
try:
    model = PipelineModel.load(MODEL_PATH)
except Exception:
    print("Failed to load model:", traceback.format_exc())
    spark.stop()
    raise

# apply transform
print("Running inference...")
pred = model.transform(df)

# ---------------------------------------------------------
# Prepare for MySQL
# ---------------------------------------------------------
if "order_id" not in pred.columns:
    pred = pred.withColumn("order_id", F.expr("uuid()"))

# Select available columns of interest
desired_cols = ["order_id", "customer_id", "market", "order_region", "shipping_mode", "customer_segment", "delivery_days", "sales"]
out_cols = [c for c in desired_cols if c in pred.columns]

if "prediction" in pred.columns: out_cols.append("prediction")

# Fix for probability vector extraction
if "probability" in pred.columns:
    pred = pred.withColumn("prob_array", vector_to_array(F.col("probability")))
    pred = pred.withColumn("prob_class_1", F.col("prob_array").getItem(1))
    out_cols.append("prob_class_1")

pred_to_write = pred.select(*out_cols)
print(f"Writing columns to MySQL: {pred_to_write.columns}")

# write to MySQL (append)
try:
    # Try Spark JDBC first
    pred_to_write.write.jdbc(url=MYSQL_JDBC_URL, table=OUTPUT_PRED_TABLE, mode="append", properties=MYSQL_PROPS)
    print("Predictions written to MySQL table", OUTPUT_PRED_TABLE)
except Exception:
    print("Spark JDBC write failed (likely missing driver); fallback to pandas/mysql-connector")
    try:
        pdf = pred_to_write.toPandas()
        import mysql.connector
        
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            port=int(MYSQL_PORT),
            user=MYSQL_USER,
            password=MYSQL_PASSWORD
        )
        cur = conn.cursor()
        
        # Create DB if not exists
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DB}")
        cur.execute(f"USE {MYSQL_DB}")
        
        # Dynamic table creation
        col_defs = []
        for c in pdf.columns:
            if c == "order_id": col_defs.append("order_id VARCHAR(255)")
            elif c == "customer_id": col_defs.append("customer_id VARCHAR(255)")
            elif c == "market": col_defs.append("market VARCHAR(255)")
            elif c == "order_region": col_defs.append("order_region VARCHAR(255)")
            elif c == "shipping_mode": col_defs.append("shipping_mode VARCHAR(255)")
            elif c == "customer_segment": col_defs.append("customer_segment VARCHAR(255)")
            elif c == "delivery_days": col_defs.append("delivery_days DOUBLE")
            elif c == "sales": col_defs.append("sales DOUBLE")
            elif c == "prediction": col_defs.append("prediction DOUBLE")
            elif c == "prob_class_1": col_defs.append("prob_class_1 DOUBLE")
            else: col_defs.append(f"{c} VARCHAR(255)")
            
        create_sql = f"CREATE TABLE IF NOT EXISTS {OUTPUT_PRED_TABLE} ({', '.join(col_defs)});"
        cur.execute(create_sql)
        conn.commit()
        
        # Dynamic Insert
        cols = list(pdf.columns)
        placeholders = ",".join(["%s"] * len(cols))
        insert_sql = f"INSERT INTO {OUTPUT_PRED_TABLE} ({','.join(cols)}) VALUES ({placeholders})"
        
        # Batch insert
        vals = []
        for _, r in pdf.iterrows():
            row_val = tuple([r[c] if pd.notna(r[c]) else None for c in cols])
            vals.append(row_val)
            
        # Chunked insert to avoid packet size limits
        chunk_size = 1000
        for i in range(0, len(vals), chunk_size):
            cur.executemany(insert_sql, vals[i:i+chunk_size])
            conn.commit()
            
        cur.close()
        conn.close()
        print(f"Fallback insert done: {len(vals)} rows")
    except Exception:
        print("Fallback insert failed:", traceback.format_exc())

# ---------------------------------------------------------
# Compute Aggregates (MongoDB)
# ---------------------------------------------------------
group_col = "market" if "market" in pred.columns else "order_region"
print(f"Aggregating by {group_col}...")

agg = pred.groupBy(group_col).agg(
    F.count("*").alias("n_orders"),
    F.avg("delivery_days").alias("avg_delivery_days"),
    F.sum("sales").alias("sum_sales")
)

agg_pdf = agg.toPandas()
if not agg_pdf.empty:
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB]
        coll = db[MONGO_COLL]
        import datetime
        now = datetime.datetime.utcnow()
        docs = []
        for _, r in agg_pdf.iterrows():
            doc = {
                "ts": now,
                "group_by": group_col,
                "region_or_market": r.get(group_col),
                "n_orders": int(r.get("n_orders")) if pd.notna(r.get("n_orders")) else 0,
                "avg_delivery_days": float(r.get("avg_delivery_days")) if pd.notna(r.get("avg_delivery_days")) else None,
                "sum_sales": float(r.get("sum_sales")) if pd.notna(r.get("sum_sales")) else 0.0
            }
            if group_col == "order_region":
                doc["market"] = r.get(group_col)
            else:
                doc["market"] = r.get("market")
                
            docs.append(doc)
            
        coll.insert_many(docs)
        client.close()
        print("Aggregates inserted into MongoDB collection", MONGO_COLL)
    except Exception:
        print("MongoDB insert failed:", traceback.format_exc())

spark.stop()
