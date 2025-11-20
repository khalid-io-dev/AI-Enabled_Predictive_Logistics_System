# fastapi_ws_bridge.py
"""
FastAPI WebSocket -> TCP bridge.

- WebSocket clients connect to ws://<host>:<http_port>/ws and send JSON messages (text).
- The bridge enqueues messages and sends newline-delimited JSON to any connected TCP client on TCP_HOST:TCP_PORT.
- Use for local dev: Spark reads from TCP (socket source) at TCP_HOST:TCP_PORT.
"""

import asyncio
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from typing import Set

HTTP_HOST = "0.0.0.0"
HTTP_PORT = 8000

TCP_HOST = "0.0.0.0"
TCP_PORT = 9999

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ws-bridge")

app = FastAPI(title="WebSocket -> TCP bridge")

# asyncio queue to decouple WS producers from TCP writers
MESSAGE_QUEUE: asyncio.Queue = asyncio.Queue(maxsize=10000)

# keep track of TCP clients (asyncio.StreamWriter objects)
TCP_CLIENTS: Set[asyncio.StreamWriter] = set()
TCP_CLIENTS_LOCK = asyncio.Lock()

# WebSocket endpoint for producers / web clients
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Accept incoming websocket messages (text JSON) and push them into MESSAGE_QUEUE.
    """
    await websocket.accept()
    logger.info("WebSocket client connected")
    try:
        while True:
            text = await websocket.receive_text()
            # Validate JSON (optional)
            try:
                obj = json.loads(text)
            except Exception:
                # If not JSON, ignore or log
                logger.warning("Received non-JSON text over websocket; ignoring")
                continue
            # Put JSON string into queue (as text)
            try:
                MESSAGE_QUEUE.put_nowait(json.dumps(obj, default=str))
            except asyncio.QueueFull:
                logger.warning("MESSAGE_QUEUE full; dropping message")
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception:
        logger.exception("WebSocket handling error")

# TCP server: accepts connections from Spark and stores writer
async def tcp_client_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info("peername")
    logger.info(f"TCP client connected: {addr}")
    async with TCP_CLIENTS_LOCK:
        TCP_CLIENTS.add(writer)
    try:
        # keep socket open until client disconnects
        while True:
            data = await reader.read(100)  # if client sends data (rare), just read and ignore
            if data == b"":
                break
            await asyncio.sleep(0.1)
    except Exception:
        logger.exception("TCP client handler error")
    finally:
        logger.info(f"TCP client disconnected: {addr}")
        async with TCP_CLIENTS_LOCK:
            if writer in TCP_CLIENTS:
                TCP_CLIENTS.remove(writer)
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

# background writer: drains MESSAGE_QUEUE and writes newline-delimited JSON to each TCP client
async def tcp_writer_loop():
    heartbeat_interval = 5.0
    last_heartbeat = asyncio.get_event_loop().time()
    while True:
        try:
            # wait for a message or timeout for heartbeat
            try:
                msg = await asyncio.wait_for(MESSAGE_QUEUE.get(), timeout=heartbeat_interval)
            except asyncio.TimeoutError:
                msg = None

            # build payloads: either a message or heartbeat
            if msg is None:
                # send heartbeat to keep connections alive if needed
                payloads = None
                now = asyncio.get_event_loop().time()
                if now - last_heartbeat >= heartbeat_interval:
                    payloads = "__heartbeat__\n"
                    last_heartbeat = now
            else:
                payloads = msg + "\n"

            if payloads is not None:
                async with TCP_CLIENTS_LOCK:
                    bad_clients = []
                    for w in list(TCP_CLIENTS):
                        try:
                            w.write(payloads.encode("utf-8"))
                            await w.drain()
                        except Exception:
                            logger.exception("Failed to write to TCP client, removing")
                            bad_clients.append(w)
                    for b in bad_clients:
                        TCP_CLIENTS.discard(b)
        except Exception:
            logger.exception("Exception in tcp_writer_loop")
            await asyncio.sleep(1.0)

# startup event: launch TCP server and writer loop
@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    # start TCP server
    server = await asyncio.start_server(tcp_client_handler, host=TCP_HOST, port=TCP_PORT)
    logger.info(f"TCP bridge listening on {TCP_HOST}:{TCP_PORT}")
    loop.create_task(server.serve_forever())
    # start writer loop
    loop.create_task(tcp_writer_loop())
    logger.info("TCP writer loop started")

# simple HTTP endpoints for testing (send a test record to queue)
@app.post("/send_test")
async def send_test():
    example = {
        "order_id": f"ORD-{asyncio.get_event_loop().time()}",
        "customer_id": "CUST-1",
        "market": "EU",
        "customer_segment": "Consumer",
        "shipping_mode": "Standard Class",
        "order_date_dateorders": "2023-09-01 12:00:00",
        "shipping_date_dateorders": "2023-09-03 15:00:00",
        "days_for_shipment_scheduled": 2,
        "days_for_shipping_real": 2,
        "benefit_per_order": 10.5,
        "sales": 100.0,
        "order_item_product_price": 30.0
    }
    await MESSAGE_QUEUE.put(json.dumps(example))
    return {"sent": example}

if __name__ == "__main__":
    # run uvicorn (it will trigger startup event)
    uvicorn.run("fastapi_ws_bridge:app", host=HTTP_HOST, port=HTTP_PORT, log_level="info")
