import asyncio
import json
import struct
import base64
import cv2
import numpy as np
import yaml
import uvicorn
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from inference_handler import InferenceHandler

# Global State
connected_clients = []
handler = None
config = {}

async def broadcast_result(camera_id, image_id, score, threshold, overlay):
    """Broadcasts a processed frame and its score to all connected WebSocket clients."""
    if not connected_clients:
        return

    # Downsize large images to prevent WebSocket payload overflow (1MB limits)
    MAX_WIDTH = 1280
    if overlay.shape[1] > MAX_WIDTH:
        scale = MAX_WIDTH / overlay.shape[1]
        overlay = cv2.resize(overlay, (MAX_WIDTH, int(overlay.shape[0] * scale)))

    # Encode BGR image to Base64 JPEG
    _, buffer = cv2.imencode('.jpg', overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    
    status = "NG" if score > threshold else "OK"
    message = {
        "camera_id": camera_id,
        "image_id": image_id,
        "score": round(score, 4),
        "threshold": threshold,
        "status": status,
        "image": f"data:image/jpeg;base64,{img_b64}"
    }
    msg_str = json.dumps(message)
    # print(f"[Broadcast] Sending {image_id}, size: {len(msg_str)} bytes")
    for client in connected_clients.copy():
        try:
            await client.send_text(msg_str)
        except Exception as e:
            print(f"[WebSocket] Error sending message (Size: {len(msg_str)} bytes): {e}")
            if client in connected_clients:
                connected_clients.remove(client)

async def handle_tcp_client(reader, writer, camera_id):
    """Async task to handle the incoming TCP stream from a camera node."""
    addr = writer.get_extra_info('peername')
    print(f"[Cam {camera_id}] Connected: {addr}")
    
    received_parts = set()
    round_count = 0
    
    try:
        while True:
            # 1. Header (4 bytes)
            try:
                header_data = await reader.readexactly(4)
            except asyncio.IncompleteReadError:
                break
                
            if not header_data:
                break
            json_size = struct.unpack('>L', header_data)[0]

            # 2. JSON Metadata
            try:
                json_bytes = await reader.readexactly(json_size)
            except asyncio.IncompleteReadError:
                break
                
            if not json_bytes:
                break
            metadata = json.loads(json_bytes.decode('utf-8'))
            image_id = metadata.get("id", "unknown")
            image_size = metadata.get("size", 0)
            
            # print(f"[Cam {camera_id}] Received image_id: {image_id} (size: {image_size})")

            # Track parts and summarize on pre_crop
            if image_id == "pre_crop":
                if len(received_parts) > 0 or round_count > 0:
                    round_count += 1
                    missing = {'crop_1', 'crop_2', 'crop_3', 'crop_4', 'crop_5', 'crop_6', 'masked_surface'} - received_parts
                    
                    print(f"\n--- [Cam {camera_id}] ROUND {round_count} SUMMARY ---")
                    print(f"   Received ({len(received_parts)}/7): {sorted(list(received_parts))}")
                    if missing:
                        print(f"   ⚠️ MISSING: {sorted(list(missing))}")
                    print(f"----------------------------------")
                    
                received_parts.clear()
            else:
                received_parts.add(image_id)

            # 3. JPEG Image Data
            if image_size <= 0:
                continue
                
            try:
                jpeg_data = await reader.readexactly(image_size)
            except asyncio.IncompleteReadError:
                break
                
            if not jpeg_data:
                break

            # Decode JPEG
            np_arr = np.frombuffer(jpeg_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            if handler is None:
                continue

            score = 0.0
            threshold = 0.0
            overlay = None

            if image_id == "pre_crop":
                # Bypass inference for pre_crop, simply send the base image
                overlay = img
            else:
                # Run inference in a background thread to prevent blocking the async loop
                score, threshold, overlay = await asyncio.to_thread(
                    handler.run_inference, image_id, img
                )
                
            if overlay is not None:
                # Broadcast the result
                await broadcast_result(camera_id, image_id, score, threshold, overlay)

    except Exception as e:
        print(f"[Cam {camera_id}] Stream error: {e}")
    finally:
        writer.close()
        await writer.wait_closed()
        print(f"[Cam {camera_id}] Connection closed.")

async def start_tcp_server(port, camera_id):
    """Starts an Asyncio TCP server for a single camera."""
    server = await asyncio.start_server(
        lambda r, w: handle_tcp_client(r, w, camera_id),
        host="0.0.0.0",
        port=port
    )
    print(f"TCP server for Cam {camera_id} listening on {port}...")
    async with server:
        await server.serve_forever()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global handler, config
    
    # Load Configuration
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Initialize Inference Handler
    handler = InferenceHandler(config_path)

    # Start TCP Receivers as background tasks
    tcp_tasks = []
    for cam_name, cam_cfg in config.get("tcp_receivers", {}).items():
        task = asyncio.create_task(
            start_tcp_server(cam_cfg["port"], cam_cfg["id"])
        )
        tcp_tasks.append(task)
        
    yield
    
    # Clean up tasks on shutdown
    for task in tcp_tasks:
        task.cancel()

app = FastAPI(lifespan=lifespan)

# Mount static files (Dashboard HTML/JS/CSS)
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def get_dashboard():
    with open(f"{static_dir}/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

if __name__ == "__main__":
    host = config.get("server", {}).get("host", "0.0.0.0") if config else "0.0.0.0"
    port = config.get("server", {}).get("port", 8000) if config else 8000
    uvicorn.run("main:app", host=host, port=port, reload=False)
