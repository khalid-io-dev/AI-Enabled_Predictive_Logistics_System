"""
Airflow DAG to orchestrate DataCo batch pipeline:
 - fetch data (from a repo / API / S3)
 - run Spark preprocessing (parquet output)
 - train model (save PipelineModel)
 - run batch prediction & persist (Postgres + Mongo)
 - optional: notify / refresh dashboard

Assumptions:
 - You have spark-submit available on the worker(s).
 - Spark jobs are provided as scripts under /opt/dataco/jobs/ (or adjust paths).
 - PostgreSQL JDBC driver path provided to Spark via --jars or placed in SPARK_CLASSPATH.
 - Airflow Connections:
    - conn_id 'postgres_default' (optional) for Postgres credentials
    - env variables for Mongo, PG or use Airflow Variables/Connections as preferred.
 - Airflow 2.x environment.

Place this file in <AIRFLOW_HOME>/dags/dataco_pipeline_dag.py
"""

from __future__ import annotations
from datetime import datetime, timedelta
import os
from pathlib import Path
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.hooks.base import BaseHook
import requests
import json
import logging

log = logging.getLogger("airflow.task")

# ----------------------
# Config - customize
# ----------------------
PROJECT_DIR = "/opt/dataco"  # change to location where your scripts live
SPARK_APP_DIR = os.path.join(PROJECT_DIR, "spark_jobs")
SCRIPTS = {
    "preprocess": os.path.join(SPARK_APP_DIR, "spark_preprocess.py"),
    "train": os.path.join(SPARK_APP_DIR, "spark_train.py"),
    "predict": os.path.join(SPARK_APP_DIR, "spark_batch_persist.py"),
    "streaming": os.path.join(SPARK_APP_DIR, "spark_streaming_windowed.py")
}
# model and data paths (HDFS/S3/local)
DATA_SOURCE_URI = "file:///data/datalake/raw/dataco.csv"   # or s3://...
CLEANED_PARQUET = "file:///data/datalake/cleaned"          # output of preprocess
MODEL_PATH = "file:///data/models/pipeline_model"
PREDICTIONS_PARQUET = "file:///data/datalake/predictions"

# JDBC jar path for SparkSubmit (example)
POSTGRES_JAR = "/opt/jars/postgresql-42.6.0.jar"

# Spark master
SPARK_MASTER = "local[*]"  # or yarn, spark://..., change in production

# Web endpoints
FASTAPI_HEALTHCHECK = "http://localhost:8000/health"   # if you run FastAPI; used by sensor
STREAMLIT_REFRESH_ENDPOINT = "http://localhost:8501/refresh"  # optional

# Airflow default args
default_args = {
    "owner": "dataco",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ----------------------
# Helper functions
# ----------------------
def push_airflow_vars_to_env(**kwargs):
    """
    Example helper to set env variables for tasks from Airflow Variables or Connections.
    You can fetch connections using BaseHook.get_connection('conn_id') or use Variables.
    This function stores values in XCom so later tasks can use them if needed.
    """
    ti = kwargs["ti"]
    # Example: read Postgres connection (if present)
    try:
        pg_conn = BaseHook.get_connection("postgres_default")
        pg = {
            "host": pg_conn.host,
            "port": pg_conn.port,
            "schema": pg_conn.schema,
            "user": pg_conn.login,
            "password": pg_conn.password,
        }
        ti.xcom_push(key="pg_conn", value=pg)
        log.info("Pushed Postgres conn to XCom (pg_conn)")
    except Exception:
        log.warning("postgres_default connection not found - skipping XCom push for pg_conn")

    # You could also push Mongo URI from Airflow Variable here
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
    ti.xcom_push(key="mongo_uri", value=mongo_uri)

# ----------------------
# DAG definition
# ----------------------
with DAG(
    dag_id="dataco_batch_end2end",
    default_args=default_args,
    description="End-to-end DataCo pipeline: preprocess, train, predict, persist, visualize",
    schedule_interval="0 2 * * *",  # daily at 02:00
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["dataco", "batch", "ml"],
) as dag:

    # 1. Sanity: push config to XCom
    push_config = PythonOperator(
        task_id="push_config_to_xcom",
        python_callable=push_airflow_vars_to_env,
    )

    # 2. Optional: ensure FastAPI bridge is up (if using streaming or to push test data)
    # HttpSensor will check the health endpoint until it returns 200 or times out.
    wait_for_fastapi = HttpSensor(
        task_id="wait_for_fastapi",
        http_conn_id=None,  # if you register an Airflow HTTP connection, set the conn id here
        endpoint=FASTAPI_HEALTHCHECK,
        method="GET",
        response_check=lambda resp: resp.status_code == 200,
        poke_interval=10,
        timeout=60,
    )

    # 3. Fetch / stage data (example: git pull or curl from remote)
    # You can implement your own data retrieval script; here a placeholder that copies from a known location.
    fetch_data = BashOperator(
        task_id="fetch_data",
        bash_command=(
            # example: copy from shared mount; change to download from S3/API as needed
            f"mkdir -p /data/datalake/raw && "
            f"cp {PROJECT_DIR}/sample_data/DataCoSupplyChainDataset.csv /data/datalake/raw/dataco.csv || true"
        ),
    )

    # 4. Run Spark preprocess job: produce cleaned parquet
    spark_preprocess = SparkSubmitOperator(
        task_id="spark_preprocess",
        application=SCRIPTS["preprocess"],   # path to spark script that reads DATA_SOURCE_URI and writes CLEANED_PARQUET
        name="dataco_preprocess",
        conn_id="spark_default",
        verbose=True,
        jars=POSTGRES_JAR,    # include any jars you need
        application_args=[
            "--input", DATA_SOURCE_URI,
            "--output", CLEANED_PARQUET,
            # you can pass other args like --model-path etc.
        ],
        conf={
            "spark.master": SPARK_MASTER,
            "spark.executor.memory": "2g",
            "spark.driver.memory": "2g",
        },
    )

    # 5. Train model
    spark_train = SparkSubmitOperator(
        task_id="spark_train",
        application=SCRIPTS["train"],
        name="dataco_train",
        conn_id="spark_default",
        jars=POSTGRES_JAR,
        application_args=[
            "--input", CLEANED_PARQUET,
            "--model_out", MODEL_PATH
        ],
        conf={
            "spark.master": SPARK_MASTER,
            "spark.executor.memory": "4g",
            "spark.driver.memory": "4g",
        },
    )

    # 6. Run batch predictions & persist to Postgres + Mongo (script uses PipelineModel located at MODEL_PATH)
    spark_predict = SparkSubmitOperator(
        task_id="spark_predict_and_persist",
        application=SCRIPTS["predict"],
        name="dataco_predict",
        conn_id="spark_default",
        jars=POSTGRES_JAR,
        application_args=[
            "--input", CLEANED_PARQUET,
            "--model", MODEL_PATH,
            "--pg_table", "predictions",
        ],
        conf={
            "spark.master": SPARK_MASTER,
            "spark.executor.memory": "2g",
            "spark.driver.memory": "2g",
        },
        verbose=True
    )

    # 7. Optional: run a light python task to compute additional aggregated metrics OR trigger a dashboard refresh URL
    def refresh_dashboard(**context):
        try:
            url = os.environ.get("STREAMLIT_REFRESH_ENDPOINT", STREAMLIT_REFRESH_ENDPOINT)
            if not url:
                log.info("No STREAMLIT_REFRESH_ENDPOINT configured; skipping refresh.")
                return
            resp = requests.post(url, timeout=5)
            log.info("Refresh response: %s %s", resp.status_code, resp.text)
        except Exception:
            log.exception("Failed to call dashboard refresh endpoint.")

    refresh_ui = PythonOperator(
        task_id="refresh_dashboard",
        python_callable=refresh_dashboard,
    )

    # 8. Cleanup / archive raw files
    cleanup = BashOperator(
        task_id="cleanup_raw",
        bash_command=(
            "mkdir -p /data/datalake/archive && "
            "mv /data/datalake/raw/dataco.csv /data/datalake/archive/dataco_$(date +%Y%m%d_%H%M%S).csv || true"
        ),
    )

    # 9. Notify (placeholder - email, slack, etc. can be used)
    notify = BashOperator(
        task_id="notify_finish",
        bash_command=f'echo "DataCo DAG complete: $(date)"'
    )

    # ----------------------
    # Define task ordering
    # ----------------------
    push_config >> wait_for_fastapi >> fetch_data >> spark_preprocess >> spark_train >> spark_predict >> refresh_ui >> cleanup >> notify

    # Optional: if you don't run FastAPI, you can skip the wait_for_fastapi:
    # push_config >> fetch_data >> spark_preprocess >> spark_train >> spark_predict >> refresh_ui >> cleanup >> notify

