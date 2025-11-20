# spark_streaming_windowed.py
"""
Spark Structured Streaming job that reads newline JSON from a TCP bridge,
parses into columns, uses windowing + watermark for aggregations,
and applies a persisted PipelineModel in foreachBatch for predictions.

Switched to MySQL for persistence.
"""

import os
import sys
import traceback
from pathlib import Path

# Ensure Spark workers use the same Python interpreter as driver (helps avoid worker crashes)
PY = sys.executable
os.environ["PYSPARK_PYTHON"] = PY
os.environ["PYSPARK_DRIVER_PYTHON"] = PY
os.environ["SPARK_PYTHON"] = PY

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.functions import vector_to_array

# for fallbacks & mongo writes
import pandas as pd
import datetime

# -----------------------
# Config
# -----------------------
# Absolute paths
BASE_DIR = r"C:\Users\ADMIN\Desktop\BRIEF-7"
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "pipeline_model"))
OUTPUT_PRED_PARQUET = os.environ.get("OUTPUT_PRED_PARQUET", os.path.join(BASE_DIR, "stream_predictions_parquet"))
OUTPUT_AGG_PARQUET = os.environ.get("OUTPUT_AGG_PARQUET", os.path.join(BASE_DIR, "window_aggregates_parquet"))
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", os.path.join(BASE_DIR, "streaming_checkpoint_windowed"))

TCP_HOST = os.environ.get("TCP_HOST", "localhost")
TCP_PORT = int(os.environ.get("TCP_PORT", "9999"))

# DB configs (used by predict_and_save)
OUTPUT_PRED_TABLE = "predictions"
MYSQL_HOST = "127.0.0.1"
MYSQL_PORT = "3306"
MYSQL_USER = "root"
MYSQL_PASSWORD = ""
MYSQL_DB = "dataco"
MYSQL_JDBC_URL = f"jdbc:mysql://{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?useSSL=false"
MYSQL_PROPS = {"user": MYSQL_USER, "password": MYSQL_PASSWORD, "driver": "com.mysql.cj.jdbc.Driver"}

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.environ.get("MONGO_DB", "dataco_analytics")
MONGO_COLLECTION_AGG = os.environ.get("MONGO_COLLECTION_AGG", "window_aggregates")

# Make directory for checkpoints
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_PRED_PARQUET).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_AGG_PARQUET).mkdir(parents=True, exist_ok=True)

# -----------------------
# Spark session (debug-friendly)
# -----------------------
spark = (
    SparkSession.builder
    .appName("DataCo_Streaming_Windowed")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "4")
    # helpful for Python worker crash tracebacks
    .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
    .config("spark.python.worker.faulthandler.enabled", "true")
    # disable arrow in streaming to avoid unexpected native issues
    .config("spark.sql.execution.arrow.pyspark.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")
print("Spark session ready. Python executable:", PY)

# -----------------------
# Read socket text lines (one JSON object per line)
# -----------------------
raw = (
    spark.readStream.format("socket")
    .option("host", TCP_HOST)
    .option("port", TCP_PORT)
    .option("includeTimestamp", "false")
    .load()
)  # single column "value" (string)

# -----------------------
# Schema for incoming JSON
# -----------------------
schema = T.StructType(
    [
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
    ]
)

# Parse JSON into columns (safe parsing)
parsed = (
    raw.select(F.col("value"))
    .withColumn("json_data", F.from_json(F.col("value"), schema))
    .select("value", "json_data.*")
)

# -----------------------
# Tolerant timestamp parsing helper
# -----------------------
def tolerant_timestamps(df):
    """Try a few timestamp formats and coalesce into order_ts and ship_ts,
       produce order_date, shipping_date, delivery_days and ingest_ts."""
    df2 = df.withColumn("order_date_dateorders", F.trim(F.col("order_date_dateorders"))) \
            .withColumn("shipping_date_dateorders", F.trim(F.col("shipping_date_dateorders")))

    # try multiple parse patterns for order_ts
    df2 = df2.withColumn("order_ts", F.to_timestamp("order_date_dateorders", "M/d/yyyy H:mm:ss"))
    df2 = df2.withColumn("order_ts", F.coalesce(F.col("order_ts"), F.to_timestamp("order_date_dateorders", "yyyy-MM-dd HH:mm:ss")))
    df2 = df2.withColumn("order_ts", F.coalesce(F.col("order_ts"), F.to_timestamp("order_date_dateorders", "M/d/yyyy H:mm")))
    df2 = df2.withColumn("order_ts", F.coalesce(F.col("order_ts"), F.to_timestamp("order_date_dateorders", "d/M/yyyy H:mm:ss")))

    # ship_ts
    df2 = df2.withColumn("ship_ts", F.to_timestamp("shipping_date_dateorders", "M/d/yyyy H:mm:ss"))
    df2 = df2.withColumn("ship_ts", F.coalesce(F.col("ship_ts"), F.to_timestamp("shipping_date_dateorders", "yyyy-MM-dd HH:mm:ss")))
    df2 = df2.withColumn("ship_ts", F.coalesce(F.col("ship_ts"), F.to_timestamp("shipping_date_dateorders", "M/d/yyyy H:mm")))
    df2 = df2.withColumn("ship_ts", F.coalesce(F.col("ship_ts"), F.to_timestamp("shipping_date_dateorders", "d/M/yyyy H:mm:ss")))

    df2 = df2.withColumn("order_date", F.to_date("order_ts")).withColumn("shipping_date", F.to_date("ship_ts"))
    df2 = df2.withColumn("delivery_days", F.datediff("shipping_date", "order_date"))
    # ingest_ts: prefer order_ts if present, else ship_ts, else current timestamp
    df2 = df2.withColumn("ingest_ts", F.coalesce(F.col("order_ts"), F.col("ship_ts"), F.current_timestamp()))
    return df2


enriched = tolerant_timestamps(parsed)

# -----------------------
# Windowed aggregation (watermark + window)
# -----------------------
WATERMARK_DELAY = "30 seconds"
WINDOW_DURATION = "1 minute"

windowed = (
    enriched.withWatermark("ingest_ts", WATERMARK_DELAY)
    .groupBy(F.window("ingest_ts", WINDOW_DURATION), F.col("market"))
    .agg(
        F.count("*").alias("n_events"),
        F.avg("delivery_days").alias("avg_delivery_days"),
        F.avg("sales").alias("avg_sales"),
    )
)


def write_window_batch(batch_df, batch_id):
    """Write windowed aggregates to parquet with window_start / window_end columns."""
    try:
        if batch_df.rdd.isEmpty():
            print(f"[window batch {batch_id}] empty; skipping")
            return
        out = batch_df.withColumn("window_start", F.col("window.start")).withColumn("window_end", F.col("window.end")).drop("window")
        # write append parquet (durable)
        out.write.mode("append").parquet(OUTPUT_AGG_PARQUET)
        print(f"[window batch {batch_id}] wrote {out.count()} rows to {OUTPUT_AGG_PARQUET}")
    except Exception:
        print("Exception in write_window_batch:")
        print(traceback.format_exc())


# Start windowed aggregation streaming
window_query = (
    windowed.writeStream.foreachBatch(write_window_batch)
    .option("checkpointLocation", os.path.join(CHECKPOINT_DIR, "window"))
    .trigger(processingTime="30 seconds")
    .start()
)
print("Windowed aggregation started")


# -----------------------
# Prediction foreachBatch
# -----------------------
# global model holder
_global = {"model": None}


def predict_and_save(batch_df, batch_id):
    """
    foreachBatch handler:
    - apply pipeline model
    - write row-level predictions to MySQL (JDBC) OR to parquet fallback
    - compute simple aggregates and insert them into MongoDB
    """
    try:
        if batch_df.rdd.isEmpty():
            print(f"[predict batch {batch_id}] empty, skipping")
            return

        # load model once
        if _global["model"] is None:
            print("Loading PipelineModel from", MODEL_PATH)
            _global["model"] = PipelineModel.load(MODEL_PATH)
            print("Model loaded")

        model = _global["model"]

        # cast numeric types as the model expects
        cast_map = {
            "benefit_per_order": "double",
            "sales": "double",
            "order_item_product_price": "double",
            "days_for_shipment_scheduled": "int",
            "days_for_shipping_real": "int",
            "delivery_days": "int",
        }
        for c, t in cast_map.items():
            if c in batch_df.columns:
                batch_df = batch_df.withColumn(c, F.col(c).cast(t))

        # transform
        pred_df = model.transform(batch_df)

        # prepare selection for writing
        selected = []
        for c in ("order_id", "customer_id", "market", "customer_segment", "shipping_mode", "delivery_days"):
            if c in pred_df.columns:
                selected.append(c)

        if "prediction" in pred_df.columns:
            selected.append("prediction")

        # extract prob_class_1 if exists
        if "probability" in pred_df.columns:
            pred_df = pred_df.withColumn(
                "prob_class_1",
                F.when(F.size(F.col("probability")) > 1, F.col("probability").getItem(1)).otherwise(F.lit(None)),
            )
            selected.append("prob_class_1")

        writable_df = pred_df.select(*selected)

        # Also write predictions to parquet as backup
        try:
            writable_df.write.mode("append").parquet(OUTPUT_PRED_PARQUET)
            print(f"[predict batch {batch_id}] appended predictions to parquet {OUTPUT_PRED_PARQUET}")
        except Exception:
            print("Failed to write predictions to parquet (non-fatal):")
            print(traceback.format_exc())

        # attempt to write to MySQL via JDBC
        try:
            writable_df.write.jdbc(url=MYSQL_JDBC_URL, table=OUTPUT_PRED_TABLE, mode="append", properties=MYSQL_PROPS)
            print(f"[predict batch {batch_id}] wrote {writable_df.count()} rows to MySQL table {OUTPUT_PRED_TABLE} via JDBC")
        except Exception:
            print("Spark JDBC write failed; falling back to pandas/mysql-connector insert. See trace:")
            print(traceback.format_exc())
            # fallback: collect small batch to Pandas and insert
            try:
                rows_pdf = writable_df.toPandas()
                if not rows_pdf.empty:
                    import mysql.connector

                    conn = None
                    try:
                        conn = mysql.connector.connect(host=MYSQL_HOST, port=int(MYSQL_PORT), user=MYSQL_USER, password=MYSQL_PASSWORD)
                        cur = conn.cursor()
                        cur.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DB}")
                        cur.execute(f"USE {MYSQL_DB}")
                        
                        # Dynamic table creation
                        col_defs = []
                        for c in rows_pdf.columns:
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

                        cols = list(rows_pdf.columns)
                        placeholders = ",".join(["%s"] * len(cols))
                        insert_sql = f"INSERT INTO {OUTPUT_PRED_TABLE} ({','.join(cols)}) VALUES ({placeholders})"
                        
                        vals = []
                        for _, r in rows_pdf.iterrows():
                            vals.append(tuple([r[c] if pd.notna(r[c]) else None for c in cols]))
                        
                        chunk_size = 1000
                        for i in range(0, len(vals), chunk_size):
                            cur.executemany(insert_sql, vals[i:i+chunk_size])
                            conn.commit()
                        cur.close()
                        print(f"[predict batch {batch_id}] fallback inserted {len(rows_pdf)} rows into MySQL via mysql-connector")
                    except Exception:
                        print("mysql-connector fallback failed:")
                        print(traceback.format_exc())
                    finally:
                        if conn:
                            conn.close()
            except Exception:
                print("Collect to pandas or mysql-connector fallback failed:")
                print(traceback.format_exc())

        # compute per-batch aggregates and write to MongoDB
        try:
            agg_df = pred_df.groupBy("market").agg(
                F.count("*").alias("n_orders"),
                F.avg("delivery_days").alias("avg_delivery_days"),
                F.sum("sales").alias("sum_sales"),
            )
            agg_pdf = agg_df.toPandas()
            if not agg_pdf.empty:
                try:
                    from pymongo import MongoClient

                    client = MongoClient(MONGO_URI)
                    db = client[MONGO_DB]
                    coll = db[MONGO_COLLECTION_AGG]
                    docs = []
                    ts = datetime.datetime.utcnow()
                    for _, r in agg_pdf.iterrows():
                        docs.append(
                            {
                                "ts": ts,
                                "batch_id": int(batch_id),
                                "market": r.get("market"),
                                "n_orders": int(r.get("n_orders")) if pd.notna(r.get("n_orders")) else 0,
                                "avg_delivery_days": float(r.get("avg_delivery_days")) if pd.notna(r.get("avg_delivery_days")) else None,
                                "sum_sales": float(r.get("sum_sales")) if pd.notna(r.get("sum_sales")) else 0.0,
                            }
                        )
                    if docs:
                        coll.insert_many(docs)
                        print(f"[predict batch {batch_id}] inserted {len(docs)} aggregate docs into MongoDB collection {MONGO_COLLECTION_AGG}")
                    client.close()
                except Exception:
                    print("MongoDB insert failed:")
                    print(traceback.format_exc())
        except Exception:
            print("Aggregate compute/write failed:")
            print(traceback.format_exc())

    except Exception:
        print("Exception in predict_and_save:")
        print(traceback.format_exc())


# start prediction streaming (foreachBatch)
prediction_query = (
    enriched.writeStream.foreachBatch(predict_and_save)
    .option("checkpointLocation", os.path.join(CHECKPOINT_DIR, "predict"))
    .trigger(processingTime="30 seconds")
    .start()
)
print("Prediction streaming started")

# Wait for termination of any stream
try:
    spark.streams.awaitAnyTermination()
except KeyboardInterrupt:
    print("Interrupted by user - stopping streams")
    for q in spark.streams.active:
        q.stop()
    spark.stop()
