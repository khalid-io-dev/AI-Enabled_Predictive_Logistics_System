# spark_streaming_job.py
"""
Spark Structured Streaming job:
- reads newline-delimited JSON from a TCP socket (host:port)
- parses JSON into columns according to a schema
- runs a foreachBatch handler that loads the saved PipelineModel (once)
  and applies pipeline_model.transform(batch_df) to produce predictions
- writes predictions to parquet for downstream consumption (or console)
- writes predictions to MySQL (fallback to Python if JDBC fails)
"""

import os
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.functions import vector_to_array
import traceback
import pandas as pd

# Absolute paths
BASE_DIR = r"C:\Users\ADMIN\Desktop\BRIEF-7"
MODEL_PATH = os.path.join(BASE_DIR, "pipeline_model")
OUTPUT_PATH = os.path.join(BASE_DIR, "stream_predictions")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "streaming_checkpoint")

HOST = "localhost"
PORT = 9999

# Config - MySQL
OUTPUT_PRED_TABLE = "predictions"
MYSQL_HOST = "127.0.0.1"
MYSQL_PORT = "3306"
MYSQL_USER = "root"
MYSQL_PASSWORD = ""
MYSQL_DB = "dataco"
MYSQL_JDBC_URL = f"jdbc:mysql://{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?useSSL=false"
MYSQL_PROPS = {"user": MYSQL_USER, "password": MYSQL_PASSWORD, "driver": "com.mysql.cj.jdbc.Driver"}

# Build Spark
spark = (SparkSession.builder
         .appName("DataCo_Streaming_Predict")
         .master("local[*]")
         .config("spark.sql.shuffle.partitions", "2")
         .config("spark.sql.execution.arrow.pyspark.enabled", "false")
         .getOrCreate())

spark.sparkContext.setLogLevel("WARN")
print("Spark session created")

# Schema: list fields your saved pipeline expects as raw input (subset)
stream_schema = T.StructType([
    T.StructField("order_id", T.StringType(), True),
    T.StructField("customer_id", T.StringType(), True),
    T.StructField("market", T.StringType(), True),
    T.StructField("customer_segment", T.StringType(), True),
    T.StructField("shipping_mode", T.StringType(), True),
    T.StructField("order_date_dateorders", T.StringType(), True),
    T.StructField("shipping_date_dateorders", T.StringType(), True),
    T.StructField("days_for_shipment_scheduled", T.IntegerType(), True),
    T.StructField("days_for_shipping_real", T.IntegerType(), True),
    T.StructField("benefit_per_order", T.DoubleType(), True),
    T.StructField("sales", T.DoubleType(), True),
    T.StructField("order_item_product_price", T.DoubleType(), True),
])

# Read raw lines from socket
raw = (spark.readStream
       .format("socket")
       .option("host", HOST)
       .option("port", PORT)
       .load())  # one string column "value"

# parse JSON: value column contains JSON string -> parse into columns
json_col = F.from_json(F.col("value"), stream_schema)
parsed = raw.select(F.col("value"), F.col("value").cast("string"), json_col.alias("data")).select("value", "data.*")

# tolerant parsing: parse timestamp formats as in batch pipeline
def enrich_batch(df):
    # replicate the feature engineering you used in batch (lightweight)
    df2 = df.withColumn("order_date_dateorders", F.trim(F.col("order_date_dateorders"))) \
            .withColumn("shipping_date_dateorders", F.trim(F.col("shipping_date_dateorders")))
    # parse timestamps with multiple formats: here we try two common ones and fallback
    # you can extend formats as needed
    df2 = df2.withColumn("order_ts", F.to_timestamp("order_date_dateorders", "M/d/yyyy H:mm:ss")) \
             .withColumn("order_ts", F.coalesce(F.col("order_ts"), F.to_timestamp("order_date_dateorders", "yyyy-MM-dd HH:mm:ss"))) \
             .withColumn("shipping_ts", F.to_timestamp("shipping_date_dateorders", "M/d/yyyy H:mm:ss")) \
             .withColumn("shipping_ts", F.coalesce(F.col("shipping_ts"), F.to_timestamp("shipping_date_dateorders", "yyyy-MM-dd HH:mm:ss"))) \
             .withColumn("order_date", F.to_date("order_ts")) \
             .withColumn("shipping_date", F.to_date("shipping_ts"))
    # delivery_days
    df2 = df2.withColumn("delivery_days", F.datediff("shipping_date", "order_date"))
    # create label if needed (not necessary for prediction)
    return df2

enriched = enrich_batch(parsed)

# foreachBatch handler — applies pipeline model to each micro-batch
# We load the pipeline model once and reuse it
_global = {"model": None}
def foreach_batch_function(batch_df, batch_id):
    try:
        if batch_df.rdd.isEmpty():
            print(f"[batch {batch_id}] empty, skipping")
            return
        # load model if not loaded
        if _global["model"] is None:
            print("Loading PipelineModel from", MODEL_PATH)
            _global["model"] = PipelineModel.load(MODEL_PATH)
            print("Model loaded")
        model = _global["model"]
        # apply any batch-level preprocessing identical to training (if necessary)
        # Here we assume the 'enriched' step produced the same raw columns the model expects
        batch_df = batch_df.withColumn("benefit_per_order", F.col("benefit_per_order").cast("double")) \
                           .withColumn("sales", F.col("sales").cast("double")) \
                           .withColumn("order_item_product_price", F.col("order_item_product_price").cast("double"))
        # perform transform (this returns a DataFrame with prediction columns)
        pred_df = model.transform(batch_df)
        # select columns to persist
        out_cols = [c for c in ("order_id", "customer_id", "market", "customer_segment", "shipping_mode", "delivery_days")]
        # ensure out cols present
        out_cols_present = [c for c in out_cols if c in pred_df.columns]
        # also include prediction and probability if present
        extra = []
        if "prediction" in pred_df.columns:
            extra.append("prediction")
        if "probability" in pred_df.columns:
            # get prob for class 1 if vector
            pred_df = pred_df.withColumn("prob_array", vector_to_array(F.col("probability")))
            pred_df = pred_df.withColumn("prob_class_1", F.col("prob_array").getItem(1))
            extra.append("prob_class_1")
        final_cols = out_cols_present + extra
        
        # write to parquet (append) or console for debugging
        print(f"[batch {batch_id}] writing {len(final_cols)} columns to {OUTPUT_PATH}")
        writable_df = pred_df.select(*final_cols)
        writable_df.write.mode("append").parquet(OUTPUT_PATH)
        print(f"[batch {batch_id}] finished writing {OUTPUT_PATH}")
        
        # Write to MySQL
        try:
            writable_df.write.jdbc(url=MYSQL_JDBC_URL, table=OUTPUT_PRED_TABLE, mode="append", properties=MYSQL_PROPS)
            print(f"[batch {batch_id}] wrote to MySQL")
        except Exception:
            print(f"[batch {batch_id}] Spark JDBC failed, falling back to Python MySQL")
            try:
                pdf = writable_df.toPandas()
                import mysql.connector
                conn = mysql.connector.connect(host=MYSQL_HOST, port=int(MYSQL_PORT), user=MYSQL_USER, password=MYSQL_PASSWORD)
                cur = conn.cursor()
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
                
                cols = list(pdf.columns)
                placeholders = ",".join(["%s"] * len(cols))
                insert_sql = f"INSERT INTO {OUTPUT_PRED_TABLE} ({','.join(cols)}) VALUES ({placeholders})"
                
                vals = []
                for _, r in pdf.iterrows():
                    vals.append(tuple([r[c] if pd.notna(r[c]) else None for c in cols]))
                
                chunk_size = 1000
                for i in range(0, len(vals), chunk_size):
                    cur.executemany(insert_sql, vals[i:i+chunk_size])
                    conn.commit()
                cur.close()
                conn.close()
                print(f"[batch {batch_id}] Python MySQL fallback success")
            except Exception:
                print(f"[batch {batch_id}] Python MySQL fallback failed: {traceback.format_exc()}")

    except Exception:
        print("Exception in foreachBatch:")
        traceback.print_exc()

# start streaming query using foreachBatch
query = (enriched.writeStream
         .foreachBatch(foreach_batch_function)
         .option("checkpointLocation", CHECKPOINT_DIR)
         .start())

print("Streaming query started, waiting for termination. Connect FastAPI producer to", HOST, PORT)
try:
    query.awaitTermination()
except KeyboardInterrupt:
    print("Stopping stream...")
    query.stop()
    spark.stop()
