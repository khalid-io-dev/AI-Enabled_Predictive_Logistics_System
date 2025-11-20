# app.py
"""
Streamlit app to serve a saved PySpark PipelineModel (classification).
- Loads PipelineModel from disk
- Infers assembler inputs and builds a friendly input form
- Single-row prediction + batch CSV upload for predictions
- Writes debug tracebacks to last_traceback.txt on failure
Notes:
 - Make sure Streamlit runs in the same Python env as PySpark.
 - On Windows prefer Python 3.11 and a clean venv to avoid worker crashes.
"""

import os
import sys
import traceback
from pathlib import Path
import tempfile

import streamlit as st

# Ensure Spark uses the same Python executable
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
os.environ.setdefault("SPARK_PYTHON", sys.executable)

# Debug trace file
TRACE_FILE = Path.cwd() / "last_traceback.txt"

def save_trace(t):
    try:
        TRACE_FILE.write_text(t, encoding="utf-8")
    except Exception:
        print("Unable to write traceback file:", traceback.format_exc())

st.set_page_config(page_title="Model Serve — DataCo", layout="wide")
st.title("DataCo — Model Serving (Spark PipelineModel)")

# 1) Start Spark
from pyspark.sql import SparkSession, Row
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F
import pandas as pd

@st.cache_resource
def create_spark():
    try:
        spark = (
            SparkSession.builder
            .appName("DataCo_ModelServe")
            .master("local[*]")
            .config("spark.executorEnv.PYSPARK_PYTHON", sys.executable)
            .config("spark.python.worker.reuse", "false")
            .config("spark.python.worker.faulthandler.enabled", "true")
            .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
            .config("spark.sql.execution.arrow.pyspark.enabled", "false")
            .getOrCreate()
        )
        return spark, None
    except Exception:
        tb = traceback.format_exc()
        save_trace(tb)
        return None, tb

spark, spark_err = create_spark()
if spark is None:
    st.error("Failed to create SparkSession. See last_traceback.txt or terminal.")
    st.stop()

st.write("Spark ready (local[*]). Python:", sys.executable)

# 2) Model load controls
st.sidebar.header("Model settings")
model_path_input = st.sidebar.text_input("Model path (local)", "./pipeline_model")
use_load = st.sidebar.button("Load pipeline model")

if "pipeline_model" not in st.session_state:
    st.session_state.pipeline_model = None
    st.session_state.assembler_inputs = None

def load_pipeline(path):
    try:
        pm = PipelineModel.load(path)
        # find VectorAssembler inputCols
        assembler_inputs = None
        for s in pm.stages:
            if isinstance(s, VectorAssembler):
                assembler_inputs = s.getInputCols()
                break
        if assembler_inputs is None:
            # try nested stages
            for s in pm.stages:
                if hasattr(s, "stages"):
                    for sub in s.stages:
                        if isinstance(sub, VectorAssembler):
                            assembler_inputs = sub.getInputCols()
                            break
                    if assembler_inputs:
                        break
        return pm, assembler_inputs, None
    except Exception:
        tb = traceback.format_exc()
        save_trace(tb)
        return None, None, tb

if use_load or st.session_state.pipeline_model is None:
    st.sidebar.info("Loading model...")
    pm, assembler_inputs, err = load_pipeline(model_path_input)
    if err:
        st.sidebar.error("Model load failed. See last_traceback.txt")
        st.error("Model load failed — traceback written to last_traceback.txt")
        st.stop()
    else:
        st.session_state.pipeline_model = pm
        st.session_state.assembler_inputs = assembler_inputs
        st.sidebar.success("Model loaded")
        st.sidebar.write(f"Assembler cols ({len(assembler_inputs) if assembler_inputs else 0}):")
        st.sidebar.write(assembler_inputs)

pipeline_model = st.session_state.pipeline_model
assembler_inputs = st.session_state.assembler_inputs
if pipeline_model is None or assembler_inputs is None:
    st.warning("No PipelineModel loaded. Provide path and click 'Load pipeline model'.")
    st.stop()

# infer raw input field names (strip _ohe/_idx suffixes)
def infer_raw_fields(cols):
    raw = []
    for c in cols:
        if c.endswith("_ohe") or c.endswith("_idx"):
            base = c.rsplit("_", 1)[0]
            raw.append(base)
        else:
            raw.append(c)
    # dedupe preserve order
    seen = set(); out = []
    for x in raw:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

raw_fields = infer_raw_fields(assembler_inputs)

st.write("Inferred input fields (fill values for prediction):")
st.write(raw_fields)

# heuristics for widget types
def guess_widget(col):
    n = col.lower()
    if any(k in n for k in ("has_", "has", "is_", "active", "flag")):
        return "select", [0,1], 1
    if any(k in n for k in ("gender", "sex")):
        return "select", ["Male","Female","Other"], "Male"
    if any(k in n for k in ("geography","country","market","city","state","region")):
        return "text", None, "unknown"
    if any(k in n for k in ("price","balance","amount","sales","salary","benefit","total")):
        return "float", None, 0.0
    if any(k in n for k in ("age","num","count","quantity","tenure","days","orders")):
        return "int", None, 1
    return "text", None, "unknown"

# 3) Single-row input form
st.header("Single-row prediction")
with st.form("single_form"):
    inputs = {}
    cols = st.columns(2)
    for i, f in enumerate(raw_fields):
        widget, opts, default = guess_widget(f)
        target = cols[i%2]
        if widget == "select":
            inputs[f] = target.selectbox(f, options=opts, index=0)
        elif widget == "int":
            inputs[f] = int(target.number_input(f, value=int(default), step=1))
        elif widget == "float":
            inputs[f] = float(target.number_input(f, value=float(default)))
        else:
            inputs[f] = target.text_input(f, value=str(default))
    submit = st.form_submit_button("Predict single row")

if submit:
    # coerce values
    def coerce(v):
        if v is None:
            return None
        if isinstance(v,(int,float)):
            return v
        s = str(v).strip()
        try:
            if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
                return int(s)
        except:
            pass
        try:
            return float(s)
        except:
            return s
    coerced = {k:coerce(v) for k,v in inputs.items()}
    st.write("Input (coerced):")
    st.json(coerced)
    try:
        row = Row(**coerced)
        input_df = spark.createDataFrame([row])
        # predict
        res_df = pipeline_model.transform(input_df).limit(1)
        rows = res_df.collect()
        if not rows:
            st.error("Model returned no rows. Check that input names/types match training schema.")
        else:
            out = rows[0].asDict(recursive=True)
            st.success("Prediction complete.")
            st.subheader("Transformed output (debug)")
            st.json(out)
            pred = out.get("prediction")
            prob = out.get("probability")
            if pred is not None:
                st.write("Prediction label:", int(pred))
            if prob is not None:
                try:
                    if hasattr(prob, "toArray"):
                        p = prob.toArray().tolist()
                    else:
                        p = list(prob)
                    st.write("Probability vector:", p)
                    if len(p) >= 2:
                        st.write(f"P(class=1) = {p[1]:.2%}")
                except Exception:
                    st.write("Probability (raw):", prob)
    except Exception:
        tb = traceback.format_exc()
        save_trace(tb)
        st.error("Prediction failed. Trace saved to last_traceback.txt")
        st.stop()

# 4) Batch CSV upload + predict
st.header("Batch prediction (CSV upload)")
st.write("Upload a CSV with the raw columns the pipeline expects (raw feature names). The pipeline will handle indexing/one-hot internally.")
uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
if uploaded is not None:
    try:
        # read CSV to pandas then to Spark (lightweight)
        pdf = pd.read_csv(uploaded)
        st.write("Preview uploaded CSV (first rows):")
        st.dataframe(pdf.head(10))
        # coerce missing columns
        missing = [c for c in raw_fields if c not in pdf.columns]
        if missing:
            st.warning(f"Uploaded CSV is missing these inferred raw fields: {missing}. Rows lacking required fields will produce nulls.")
        sdf = spark.createDataFrame(pdf)
        st.write("Running pipeline.transform on uploaded file (this may take some time).")
        pred_sdf = pipeline_model.transform(sdf)
        # pick a compact set of columns to export: original + prediction + probability
        export_cols = pdf.columns.tolist()
        if "prediction" not in pred_sdf.columns:
            st.warning("Pipeline did not produce 'prediction' column. Inspect pipeline.")
        else:
            # probability may be vector; convert to arrays then extract probability_1
            from pyspark.sql.types import DoubleType
            def extract_prob1(df):
                if "probability" in df.columns:
                    return df.withColumn("prob_class_1", F.when(F.size(F.col("probability"))>1, F.col("probability").getItem(1)).otherwise(F.lit(None)))
                else:
                    return df
            pred_sdf = extract_prob1(pred_sdf)
            out_cols = export_cols + [c for c in ["prediction","prob_class_1"] if c in pred_sdf.columns]
            result_pdf = pred_sdf.select(*out_cols).toPandas()
            st.success("Batch prediction finished. Preview:")
            st.dataframe(result_pdf.head(20))
            # allow download
            csv_bytes = result_pdf.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
    except Exception:
        tb = traceback.format_exc()
        save_trace(tb)
        st.error("Batch prediction failed — traceback saved to last_traceback.txt")
        st.stop()

# 5) Save predictions to disk example (utility)
st.markdown("---")
st.write("Utility: save a small summary of model outputs (market summary) if present in the dataset.")
if st.button("Export sample market summary from a small sample (demo)"):
    try:
        # attempt to read one saved parquet or CSV in cwd sample (if any)
        demo_path = Path.cwd() / "sample_input.csv"
        if not demo_path.exists():
            st.warning("No sample_input.csv found in cwd; create one or upload CSV using the batch tool.")
        else:
            demo_pdf = pd.read_csv(demo_path)
            demo_sdf = spark.createDataFrame(demo_pdf)
            pred_demo = pipeline_model.transform(demo_sdf)
            if "market" in pred_demo.columns and "delivery_days" in pred_demo.columns:
                summary = pred_demo.groupBy("market").agg(F.count("*").alias("n_orders"), F.avg("delivery_days").alias("avg_delivery_days"), F.sum("sales").alias("sum_sales"))
                out_path = "./market_summary_export.csv"
                summary.toPandas().to_csv(out_path, index=False)
                st.success(f"Market summary exported to {out_path}")
            else:
                st.warning("Required columns 'market' or 'delivery_days' missing in sample dataset.")
    except Exception:
        tb = traceback.format_exc()
        save_trace(tb)
        st.error("Failed to export market summary; traceback saved to last_traceback.txt")

# Developer panel
with st.expander("Developer / Pipeline info"):
    st.write("Pipeline stages:")
    try:
        for i, s in enumerate(pipeline_model.stages):
            st.write(f"{i}. {type(s).__name__} (uid={getattr(s,'uid',None)})")
    except Exception:
        st.write("Could not enumerate stages:", traceback.format_exc())
    st.write("Assembler inputCols (raw):")
    st.write(assembler_inputs)
    st.write("Inferred raw fields shown to UI:")
    st.write(raw_fields)
    st.write("last_traceback.txt contents (if any):")
    if TRACE_FILE.exists():
        try:
            st.code(TRACE_FILE.read_text())
        except Exception:
            st.write("Could not read trace file.")

st.markdown(
    """
    **Notes & tips**
    - Run Streamlit in the same Python environment where `pyspark` is installed.
    - On Windows prefer Python 3.11 + clean venv to avoid "Python worker exited unexpectedly" issues.
    - Persisted model folder (./pipeline_model) should be the folder you saved with `model.write().save(...)`.
    - For production, consider wrapping predictions behind a lightweight API (FastAPI) and using a non-Spark microservice or model export (ONNX/PMML/MLflow) if low-latency predictions are required.
    """
)
