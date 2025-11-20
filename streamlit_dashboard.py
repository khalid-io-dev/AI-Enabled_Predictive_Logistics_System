# streamlit_dashboard.py
"""
Streamlit dashboard to visualize aggregated analytics stored in MongoDB.

Expected aggregated document schema used by the pipeline:
{
  "ts": <ISO datetime>,
  "batch_id": <int>,
  "market": "EU",
  "n_orders": 42,
  "avg_delivery_days": 3.2,
  "sum_sales": 12345.67
}

Usage:
    pip install streamlit pymongo pandas plotly
    streamlit run streamlit_dashboard.py

Environment / quick config:
 - You can also provide a Mongo URI in the sidebar or use MONGO_URI env var.
"""

import os
import io
import datetime as dt
from typing import Optional

import streamlit as st
import pandas as pd

# plotting
import plotly.express as px

# DB client
from pymongo import MongoClient
from pymongo.collection import Collection

st.set_page_config(page_title="DataCo Analytics Dashboard", layout="wide")

st.title("DataCo — Aggregates Dashboard")
st.markdown("Visualize aggregated streaming/batch analytics stored in MongoDB.")

# -------------------------
# Sidebar - connection + filters
# -------------------------
st.sidebar.header("Connection & Query")
MONGO_URI_DEFAULT = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
mongo_uri = st.sidebar.text_input("MongoDB URI", value=MONGO_URI_DEFAULT)
mongo_db = st.sidebar.text_input("MongoDB database", value=os.environ.get("MONGO_DB", "dataco_analytics"))
mongo_collection = st.sidebar.text_input("Collection (aggregates)", value=os.environ.get("MONGO_COLLECTION_AGG", "window_aggregates"))

# time filter defaults
now = dt.datetime.utcnow()
default_from = now - dt.timedelta(days=7)
date_from = st.sidebar.date_input("From date (UTC)", value=default_from.date())
date_to = st.sidebar.date_input("To date (UTC)", value=now.date())

# market filter
market_filter = st.sidebar.text_input("Market filter (leave blank for all)", value="")

# refresh controls
refresh = st.sidebar.button("Refresh now")
auto_refresh = st.sidebar.checkbox("Auto refresh every 60s", value=False)
st.sidebar.caption("Press 'Refresh now' after changing filters.")

# -------------------------
# DB helpers
# -------------------------
@st.cache_data(ttl=30)
def get_mongo_collection(uri: str, db: str, coll_name: str) -> Collection:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    _db = client[db]
    return _db[coll_name]

@st.cache_data(ttl=30)
def fetch_aggregates(coll: Collection, dt_from: dt.datetime, dt_to: dt.datetime, market: Optional[str] = None) -> pd.DataFrame:
    """
    Query Mongo collection and return DataFrame.
    """
    # build query
    q = {"ts": {"$gte": dt_from, "$lte": dt_to}}
    if market:
        q["market"] = market
    cursor = coll.find(q, projection={"_id": 0})
    df = pd.DataFrame(list(cursor))
    if df.empty:
        return df
    # ensure ts datetime
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.sort_values("ts")
    # coerce numeric types
    for col in ("n_orders", "avg_delivery_days", "sum_sales"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# -------------------------
# Load collection and data
# -------------------------
try:
    coll = get_mongo_collection(mongo_uri, mongo_db, mongo_collection)
except Exception as e:
    st.error("Failed to connect to MongoDB. Check URI and network. Error: " + str(e))
    st.stop()

# Build datetime bounds
dt_from = dt.datetime.combine(date_from, dt.time.min).replace(tzinfo=dt.timezone.utc)
dt_to = dt.datetime.combine(date_to, dt.time.max).replace(tzinfo=dt.timezone.utc)

# fetch
df = fetch_aggregates(coll, dt_from, dt_to, market_filter.strip() or None)

# manual refresh: clear cache then re-fetch
if refresh:
    st.experimental_rerun()

# auto refresh mechanism (very simple)
if auto_refresh:
    st.autorefresh(interval=60 * 1000, key="autorefresh")

# -------------------------
# Top-level summary / KPIs
# -------------------------
st.subheader("Summary")

if df.empty:
    st.warning("No aggregated documents found for the selected filters.")
    st.write("Query params:", {"db": mongo_db, "collection": mongo_collection, "from": dt_from.isoformat(), "to": dt_to.isoformat(), "market": market_filter})
else:
    # compute overall KPIs for the filtered set
    total_orders = int(df["n_orders"].sum()) if "n_orders" in df.columns else 0
    avg_delivery = float(df["avg_delivery_days"].mean()) if "avg_delivery_days" in df.columns else None
    total_sales = float(df["sum_sales"].sum()) if "sum_sales" in df.columns else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Orders (filtered)", f"{total_orders:,}")
    col2.metric("Avg Delivery Days", f"{avg_delivery:.2f}" if avg_delivery is not None else "N/A")
    col3.metric("Total Sales", f"{total_sales:,.2f}")

    # -------------------------
    # Time series chart: n_orders over time
    # -------------------------
    st.markdown("### Orders over time")
    if "ts" in df.columns and "n_orders" in df.columns:
        # group by ts (maybe bucket by minute/hour depending on density)
        ts_df = df.set_index("ts").resample("1H").agg({"n_orders": "sum"}).reset_index()
        fig_ts = px.line(ts_df, x="ts", y="n_orders", title="Orders over time (resampled 1H)")
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("Time series not available: missing 'ts' or 'n_orders' in documents.")

    # -------------------------
    # Bar chart: avg_delivery_days by market
    # -------------------------
    st.markdown("### Avg delivery days by market (grouped)")
    if "market" in df.columns and "avg_delivery_days" in df.columns:
        by_market = df.groupby("market").agg({"n_orders": "sum", "avg_delivery_days": "mean", "sum_sales": "sum"}).reset_index()
        fig_bar = px.bar(by_market, x="market", y="avg_delivery_days", text="n_orders",
                         title="Average delivery days by market (bubble shows orders)")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Market-level aggregates not available in documents.")

    # -------------------------
    # Table and download
    # -------------------------
    st.markdown("### Raw aggregated rows (paginated preview)")
    # show last 200 rows
    max_rows = st.slider("Max rows to show", 10, 1000, 200, step=10)
    st.dataframe(df.tail(max_rows).reset_index(drop=True))

    # CSV download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV (current filter)", data=csv, file_name="aggregates.csv", mime="text/csv")

    # -------------------------
    # Small exploratory filters
    # -------------------------
    st.markdown("### Explore by Market / Batch")
    markets = sorted(df["market"].dropna().unique().tolist()) if "market" in df.columns else []
    selected_market = st.selectbox("Select market to inspect", options=["(all)"] + markets, index=0)
    if selected_market != "(all)":
        df_market = df[df["market"] == selected_market]
        st.write(f"Showing {len(df_market)} aggregate rows for market = {selected_market}")
        st.line_chart(df_market.set_index("ts")["n_orders"].resample("1H").sum())

# -------------------------
# Footer / notes
# -------------------------
st.markdown("---")
st.caption(
    "Notes: This dashboard reads aggregated documents from MongoDB and uses Plotly for interactive charts. "
    "For production dashboards, save feature metadata and canonical time buckets when writing aggregates to MongoDB."
)
