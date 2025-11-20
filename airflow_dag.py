# airflow_dag.py
"""
Example Airflow DAG to orchestrate:
- start FastAPI producer (uvicorn) in background
- run spark-submit streaming job
- stop/cleanup

WARNING: This is a simple demo DAG. In production use Docker, systemd, or k8s for process management.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import os
import signal
import subprocess
import time

BASE_DIR = "/path/to/your/project"  # update
FASTAPI_SCRIPT = os.path.join(BASE_DIR, "fastapi_producer.py")
SPARK_SCRIPT = os.path.join(BASE_DIR, "spark_streaming_job.py")

default_args = {
    "owner": "you",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    "dataco_streaming_pipeline",
    default_args=default_args,
    description="Start producer and run Spark streaming job",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
)

start_producer = BashOperator(
    task_id="start_fastapi_producer",
    bash_command=f"nohup python {FASTAPI_SCRIPT} > {BASE_DIR}/fastapi.log 2>&1 & echo $! > {BASE_DIR}/fastapi.pid",
    dag=dag,
)

run_streaming_job = BashOperator(
    task_id="run_spark_streaming",
    bash_command=f"/path/to/spark/bin/spark-submit --master local[*] {SPARK_SCRIPT}",
    dag=dag,
)

# stop producer task
stop_producer = BashOperator(
    task_id="stop_fastapi_producer",
    bash_command="if [ -f {base}/fastapi.pid ]; then kill $(cat {base}/fastapi.pid) || true; fi".format(base=BASE_DIR),
    dag=dag,
)

start_producer >> run_streaming_job >> stop_producer
