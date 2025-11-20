# fastapi_producer.py
"""
FastAPI app that:
- runs a TCP socket server on host:port (default localhost:9999)
- when a client connects, sends newline-delimited JSON records periodically
- exposes HTTP endpoints to control the generator (start/stop/change-rate/send-one)
Usage:
    python fastapi_producer.py
Then:
    uvicorn will run FastAPI on port 8000
Spark can connect to the TCP source at host=localhost, port=9999
"""

import asyncio
import json
import random
import signal
import threading
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
from faker import Faker

HOST = "0.0.0.0"   # listening for Spark on same host
TCP_PORT = 9999
HTTP_PORT = 8000

fake = Faker()
app = FastAPI(title="DataCo Streaming Producer")

# state
_state = {
    "running": False,
    "interval_sec": 1.0,    # seconds between messages
    "clients": set(),       # set of writer streams
    "lock": threading.Lock()
}

# Example minimal schema for streaming records (pick subset used by model)
def gen_dataco_record():
    """Return a dictionary similar to a row from your dataco dataset"""
    # simple synthetic fields — expand as needed
    order_id = f"ORD{random.randint(100000,999999)}"
    customer_id = f"CUST{random.randint(10000,99999)}"
    market = random.choice(["US", "EU", "APAC", "MENA"])
    customer_segment = random.choice(["Consumer", "Corporate", "Home Office", "Small Business"])
    shipping_mode = random.choice(["Standard Class","Second Class","First Class","Same Day"])
    order_date = fake.date_time_between(start_date='-2y', end_date='now').strftime("%Y-%m-%d %H:%M:%S")
    # shipping date = order_date + some days
    ship_delay = random.choice([0,1,2,3,4,5,7,10])
    ship_dt = (datetime.strptime(order_date, "%Y-%m-%d %H:%M:%S") + timedelta(days=ship_delay)).strftime("%Y-%m-%d %H:%M:%S")
    days_for_shipment_scheduled = random.choice([1,2,3,5,7])
    days_for_shipping_real = ship_delay
    benefit_per_order = round(random.uniform(-10, 200), 2)
    sales = round(random.uniform(5, 2000), 2)
    product_price = round(random.uniform(1, 500), 2)

    return {
        "order_id": order_id,
        "customer_id": customer_id,
        "market": market,
        "customer_segment": customer_segment,
        "shipping_mode": shipping_mode,
        "order_date_dateorders": order_date,
        "shipping_date_dateorders": ship_dt,
        "days_for_shipment_scheduled": days_for_shipment_scheduled,
        "days_for_shipping_real": days_for_shipping_real,
        "benefit_per_order": benefit_per_order,
        "sales": sales,
        "order_item_product_price": product_price,
        # add any other fields needed by your saved pipeline
    }

# TCP server: accept a single connection (Spark), broadcast JSON lines
async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info("peername")
    print(f"[TCP] Client connected: {addr}")
    with _state["lock"]:
        _state["clients"].add(writer)
    try:
        # keep the connection open until closed externally
        while not writer.is_closing():
            await asyncio.sleep(1.0)
    finally:
        print(f"[TCP] Client disconnected: {addr}")
        with _state["lock"]:
            _state["clients"].discard(writer)
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

async def tcp_server_task(host, port):
    server = await asyncio.start_server(handle_client, host=host, port=port)
    print(f"[TCP] Server listening on {host}:{port}")
    async with server:
        await server.serve_forever()

# background coroutine that sends data periodically to connected writers
async def producer_loop():
    while True:
        if _state["running"] and _state["clients"]:
            rec = gen_dataco_record()
            text = json.dumps(rec, default=str) + "\n"  # newline-delimited JSON
            # write to all connected clients; remove closed ones
            with _state["lock"]:
                to_remove = []
                for w in list(_state["clients"]):
                    try:
                        w.write(text.encode("utf-8"))
                        await w.drain()
                    except Exception as e:
                        print("[TCP] client write failed:", e)
                        to_remove.append(w)
                for w in to_remove:
                    _state["clients"].discard(w)
        await asyncio.sleep(_state["interval_sec"])

# Start background tasks when FastAPI starts
@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    # start the TCP server
    loop.create_task(tcp_server_task(HOST, TCP_PORT))
    # start the producer loop
    loop.create_task(producer_loop())
    print("[FastAPI] Startup complete. TCP server and producer started.")

# Control endpoints
class ControlRequest(BaseModel):
    interval_sec: Optional[float] = None

@app.post("/start")
async def start_stream(req: ControlRequest = None):
    if req and req.interval_sec:
        _state["interval_sec"] = float(req.interval_sec)
    _state["running"] = True
    return {"status": "started", "interval_sec": _state["interval_sec"]}

@app.post("/stop")
async def stop_stream():
    _state["running"] = False
    return {"status": "stopped"}

@app.post("/set_interval")
async def set_interval(req: ControlRequest):
    if req and req.interval_sec:
        _state["interval_sec"] = float(req.interval_sec)
    return {"interval_sec": _state["interval_sec"]}

@app.post("/send_one")
async def send_one():
    rec = gen_dataco_record()
    text = json.dumps(rec, default=str) + "\n"
    with _state["lock"]:
        for w in list(_state["clients"]):
            try:
                w.write(text.encode("utf-8"))
                await w.drain()
            except Exception:
                _state["clients"].discard(w)
    return {"sent": rec}

@app.get("/status")
async def status():
    return {
        "running": _state["running"],
        "interval_sec": _state["interval_sec"],
        "clients_connected": len(_state["clients"])
    }

# Run with uvicorn when invoked directly
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI producer with TCP server...")
    uvicorn.run("fastapi_producer:app", host="0.0.0.0", port=HTTP_PORT, reload=False)
