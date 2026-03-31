import asyncio
import subprocess
import json
import struct
import base64
import cv2
import numpy as np
import yaml
import uvicorn
import os
import random
import glob
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Optional, List
from datetime import datetime
import threading
import paho.mqtt.client as mqtt

from inference_handler import InferenceHandler

# Global State
connected_clients = []
debug_clients = {}
handler = None
config = {}
NG_STATS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ng_stats.json")
stats_lock = threading.Lock()
demo_active = {}
demo_tasks = {}
mqtt_client = None
device_statuses = {}

async def broadcast_sys_status(msg_str):
    for client in connected_clients.copy():
        try:
            await client.send_text(msg_str)
        except Exception:
            if client in connected_clients:
                connected_clients.remove(client)

def on_mqtt_connect(client, userdata, flags, rc, *args):
    print(f"Connected to MQTT broker with result code {rc}")
    status_topic = config.get("mqtt", {}).get("status_topic", "+/sys/status")
    client.subscribe(status_topic)

async def broadcast_mqtt_debug(msg_str, topic):
    dead_clients = []
    for ws, subs in list(debug_clients.items()):
        for sub in subs:
            if mqtt.topic_matches_sub(sub, topic):
                try:
                    await ws.send_text(msg_str)
                except Exception:
                    dead_clients.append(ws)
                break
    for client in dead_clients:
        if client in debug_clients:
            del debug_clients[client]

def on_mqtt_message(client, userdata, msg):
    try:
        topic = msg.topic
        try:
            payload_str = msg.payload.decode('utf-8')
        except UnicodeDecodeError:
            payload_str = str(msg.payload)
            
        # --- Debug Broadcast ---
        if debug_clients and userdata and 'loop' in userdata:
            debug_msg = json.dumps({"topic": topic, "payload": payload_str})
            asyncio.run_coroutine_threadsafe(broadcast_mqtt_debug(debug_msg, topic), userdata['loop'])

        # --- System Status Tracker ---
        status_topic = config.get("mqtt", {}).get("status_topic", "+/sys/status")
        if mqtt.topic_matches_sub(status_topic, topic):
            try:
                payload = json.loads(payload_str)
                hostname = payload.get("hostname", "unknown")
                status = payload.get("system", "offline")
                
                # Update internal tracker
                device_statuses[hostname] = {
                    "status": status,
                    "last_seen": int(datetime.now().timestamp() * 1000)
                }
                
                ws_msg = {
                    "type": "sys_status",
                    "data": payload
                }
                msg_str = json.dumps(ws_msg)
                
                if userdata and 'loop' in userdata:
                    asyncio.run_coroutine_threadsafe(broadcast_sys_status(msg_str), userdata['loop'])
            except json.JSONDecodeError:
                pass # Ignore non-JSON messages for system status
    except Exception as e:
        print(f"Error processing MQTT message: {e}")

try:
    import RPi.GPIO as GPIO
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(23, GPIO.OUT)
    GPIO.output(23, GPIO.LOW)
    GPIO.setup(24, GPIO.OUT)  # Changed from IN to OUT for global NG trigger
    GPIO.output(24, GPIO.LOW)
    gpio_available = True
except ImportError:
    print("RPi.GPIO module not found. Simulating GPIO 23 and 24.")
    gpio_available = False

gpio_pulse_task = None

# Global state for Real-time GPIO monitoring
gpio_states = {
    23: {"state": 0, "last_changed": None},
    24: {"state": 0, "last_changed": None}
}

global_demo_active = False
global_demo_task = None
global_demo_interval = 5.0
global_demo_paused = False

async def trigger_gpio_pulse(camera_id):
    """Trigger a 1-second pulse on GPIO 23 and 24."""
    try:
        if gpio_available:
            GPIO.output(23, GPIO.HIGH)
            GPIO.output(24, GPIO.HIGH)
        else:
            print(f"[GPIO SIMULATION] Cam {camera_id} triggered. GPIO 23 AND 24 -> HIGH")
            
        await asyncio.sleep(1.0)
        
        if gpio_available:
            GPIO.output(23, GPIO.LOW)
            GPIO.output(24, GPIO.LOW)
        else:
            print(f"[GPIO SIMULATION] Cam {camera_id} finished. GPIO 23 AND 24 -> LOW")
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"GPIO Error: {e}")

async def broadcast_gpio_status(pin, state, last_changed):
    """Broadcasts GPIO state change to all connected WebSocket clients."""
    if not connected_clients:
        return

    message = {
        "type": "gpio",
        "pin": pin,
        "state": state,
        "last_changed": last_changed
    }
    msg_str = json.dumps(message)
    for client in connected_clients.copy():
        try:
            await client.send_text(msg_str)
        except Exception:
            if client in connected_clients:
                connected_clients.remove(client)

async def monitor_gpio_status():
    """Async loop to continuously monitor GPIO 23 & 24 for state changes."""
    # Initialize timestamps
    now_iso = datetime.now().isoformat()
    gpio_states[23]["last_changed"] = now_iso
    gpio_states[24]["last_changed"] = now_iso
    
    # Initialize actual states if available
    if gpio_available:
        gpio_states[23]["state"] = GPIO.input(23)
        gpio_states[24]["state"] = GPIO.input(24)

    while True:
        try:
            now_iso = datetime.now().isoformat()
            
            # Read GPIO 23 (Output state if simulated, input state if real GPIO output allows readback)
            # In RPi.GPIO, reading an OUT pin returns its set state.
            current_23 = GPIO.input(23) if gpio_available else gpio_states[23]["state"]
            if current_23 != gpio_states[23]["state"]:
                gpio_states[23]["state"] = current_23
                gpio_states[23]["last_changed"] = now_iso
                await broadcast_gpio_status(23, current_23, now_iso)

            # Read GPIO 24
            current_24 = GPIO.input(24) if gpio_available else gpio_states[24]["state"]
            if current_24 != gpio_states[24]["state"]:
                gpio_states[24]["state"] = current_24
                gpio_states[24]["last_changed"] = now_iso
                await broadcast_gpio_status(24, current_24, now_iso)
                
            await asyncio.sleep(0.05) # 50ms polling rate
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"GPIO Monitoring Error: {e}")
            await asyncio.sleep(1)

def load_ng_stats():
    with stats_lock:
        if os.path.exists(NG_STATS_FILE):
            try:
                with open(NG_STATS_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading NG stats: {e}")
                return {}
        return {}

def save_ng_stats(stats):
    with stats_lock:
        try:
            with open(NG_STATS_FILE, "w") as f:
                json.dump(stats, f, indent=4)
        except Exception as e:
            print(f"Error saving NG stats: {e}")

async def demo_loop(cam_id):
    global demo_active
    demo_ok_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_images", "ok")
    demo_ng_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_images", "ng")
    
    crops = ['crop_1', 'crop_2', 'crop_3', 'crop_4', 'crop_5', 'crop_6', 'masked_surface', 'pre_crop']
    
    while demo_active.get(cam_id, False):
        ok_images = glob.glob(os.path.join(demo_ok_dir, "*.jpg")) + glob.glob(os.path.join(demo_ok_dir, "*.png"))
        ng_images = glob.glob(os.path.join(demo_ng_dir, "*.jpg")) + glob.glob(os.path.join(demo_ng_dir, "*.png"))
        all_images = [(img, 'OK') for img in ok_images] + [(img, 'NG') for img in ng_images]
        
        if not all_images:
            await asyncio.sleep(2)
            continue
            
        random.shuffle(all_images)
        
        for img_path, expected_status in all_images:
            if not demo_active.get(cam_id, False):
                break
                
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Broadcast raw_image
            await broadcast_result(cam_id, 'raw_image', 0.0, 0.5, img, 0.0)
            await asyncio.sleep(0.1)
            
            # Broadcast crops
            for crop in crops:
                if not demo_active.get(cam_id, False):
                    break
                
                threshold = 0.5
                if config and 'models' in config and crop in config['models']:
                    threshold = config['models'][crop].get('threshold', 0.5)
                
                if expected_status == 'OK':
                    score = threshold * random.uniform(0.1, 0.8)
                else:
                    if random.random() > 0.5 or crop == 'crop_1':
                        score = threshold * random.uniform(1.1, 1.5)
                    else:
                        score = threshold * random.uniform(0.1, 0.8)

                h, w = img.shape[:2]
                cw, ch = w//3 or 1, h//3 or 1
                cx = random.randint(0, max(0, w - cw))
                cy = random.randint(0, max(0, h - ch))
                crop_img = img[cy:cy+ch, cx:cx+cw]
                
                await broadcast_result(cam_id, crop, score, threshold, crop_img, 0.015)
                await asyncio.sleep(0.05)
                
            # Wait before next simulated inference
            await asyncio.sleep(3)

async def global_demo_loop():
    global global_demo_active, global_demo_interval, global_demo_paused
    demo_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_images")
    
    crops = ['crop_1', 'crop_2', 'crop_3', 'crop_4', 'crop_5', 'crop_6', 'masked_surface', 'pre_crop']
    
    # 7 Cameras in 5 Physical Planes
    plane_cameras = [0, 1, 2, 3, 4, 5, 6] 
    plane_to_folder = {
        0: "plane_1",
        1: "plane_1",
        2: "plane_2",
        3: "plane_2",
        4: "plane_3",
        5: "plane_4",
        6: "plane_5"
    }
    import time
    processing_time = 2.5  # Initial estimate for a full round
    
    while global_demo_active:
        if global_demo_paused:
            await asyncio.sleep(0.5)
            continue
            
        # Deduct expected processing time so the round finishes exactly when the interval hits 0
        sleep_amount = global_demo_interval - processing_time
        if sleep_amount > 0:
            slept = 0.0
            while slept < sleep_amount and global_demo_active and not global_demo_paused:
                chunk = min(0.5, sleep_amount - slept)
                await asyncio.sleep(chunk)
                slept += chunk
                
        if not global_demo_active or global_demo_paused:
            continue
            
        t_start = time.time()
        
        # Select one random plane to be NG
        ng_plane = random.choice(plane_cameras)
        
        has_ng_in_round = False
        
        for cam_id in plane_cameras:
            if not global_demo_active:
                break
                
            is_ng = (cam_id == ng_plane)
            folder_name = plane_to_folder.get(cam_id, "plane_1")
            cam_ok_dir = os.path.join(demo_base_dir, folder_name, "ok")
            cam_ng_dir = os.path.join(demo_base_dir, folder_name, "ng")
            
            ok_images = glob.glob(os.path.join(cam_ok_dir, "*.jpg")) + glob.glob(os.path.join(cam_ok_dir, "*.png"))
            ng_images = glob.glob(os.path.join(cam_ng_dir, "*.jpg")) + glob.glob(os.path.join(cam_ng_dir, "*.png"))
            
            img_path = None
            if is_ng and ng_images:
                img_path = random.choice(ng_images)
            elif not is_ng and ok_images:
                img_path = random.choice(ok_images)
            else:
                # Fallback to general ok/ng if specific ones don't exist
                global_ok = glob.glob(os.path.join(demo_base_dir, "ok", "*.jpg")) + glob.glob(os.path.join(demo_base_dir, "ok", "*.png"))
                global_ng = glob.glob(os.path.join(demo_base_dir, "ng", "*.jpg")) + glob.glob(os.path.join(demo_base_dir, "ng", "*.png"))
                if is_ng and global_ng:
                    img_path = random.choice(global_ng)
                elif not is_ng and global_ok:
                    img_path = random.choice(global_ok)
            
            if not img_path:
                continue
                
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            # Broadcast raw_image
            await broadcast_result(cam_id, 'raw_image', 0.0, 0.5, img, 0.0)
            await asyncio.sleep(0.05)
            
            # Broadcast crops
            for crop in crops:
                if not global_demo_active:
                    break
                
                threshold = 0.5
                if config and 'models' in config and crop in config['models']:
                    threshold = config['models'][crop].get('threshold', 0.5)
                
                if not is_ng:
                    score = threshold * random.uniform(0.1, 0.8)
                else:
                    if random.random() > 0.5 or crop == 'crop_1':
                        score = threshold * random.uniform(1.1, 1.5)
                        has_ng_in_round = True
                    else:
                        score = threshold * random.uniform(0.1, 0.8)

                h, w = img.shape[:2]
                cw, ch = w//3 or 1, h//3 or 1
                cx = random.randint(0, max(0, w - cw))
                cy = random.randint(0, max(0, h - ch))
                crop_img = img[cy:cy+ch, cx:cx+cw]
                
                await broadcast_result(cam_id, crop, score, threshold, crop_img, 0.015)
                await asyncio.sleep(0.02)
                
        t_end = time.time()
        actual_pt = t_end - t_start
        # Exponential moving average to smooth variations in processing time
        processing_time = (0.5 * processing_time) + (0.5 * actual_pt)
                
        if has_ng_in_round:
            # Tell the frontend to show the NG overlay freeze dialog
            msg_str = json.dumps({"type": "global_demo_freeze"})
            for client in connected_clients.copy():
                try:
                    await client.send_text(msg_str)
                except Exception:
                    pass
            global_demo_paused = True
            
        while global_demo_paused and global_demo_active:
            await asyncio.sleep(0.5)

async def broadcast_result(camera_id, image_id, score, threshold, overlay, inference_time=0.0):
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
    
    if status == "NG":
        if config.get("gpio_triggers", {}).get(str(camera_id), False):
            global gpio_pulse_task
            if gpio_pulse_task and not gpio_pulse_task.done():
                gpio_pulse_task.cancel()
            gpio_pulse_task = asyncio.create_task(trigger_gpio_pulse(camera_id))

    message = {
        "camera_id": camera_id,
        "image_id": image_id,
        "score": round(score, 4),
        "threshold": threshold,
        "status": status,
        "inference_time": inference_time,
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
            
            print(f"[Cam {camera_id}] Received image_id: {image_id} (size: {image_size})")

            # Track parts and summarize on raw_image
            if image_id == "raw_image":
                if len(received_parts) > 0 or round_count > 0:
                    round_count += 1
                    
                    # Dynamically get expected crops for this camera
                    expected = set()
                    for cam_key, cam_val in config.get("tcp_receivers", {}).items():
                        if cam_val.get("id") == camera_id:
                            expected = set(cam_val.get("expected_crops", config.get("models", {}).keys() - {"raw_image"}))
                            break
                    if not expected:
                        expected = set(config.get("models", {}).keys()) - {"raw_image"}

                    missing = expected - received_parts
                    
                    print(f"\n--- [Cam {camera_id}] ROUND {round_count} SUMMARY ---")
                    print(f"   Received ({len(received_parts)}/{len(expected)}): {sorted(list(received_parts))}")
                    if missing:
                        print(f"   ⚠️ MISSING: {sorted(list(missing))}")
                    print(f"----------------------------------")
                    
                received_parts.clear()
            else:
                received_parts.add(image_id)

            # 3. JPEG Image Data
            if image_id == "error":
                error_msg = metadata.get("message", "Unknown Proxy Error")
                print(f"[Cam {camera_id}] Proxy Error: {error_msg}")
                error_json = json.dumps({
                    "type": "camera_error",
                    "camera_id": camera_id,
                    "message": error_msg
                })
                for client in connected_clients.copy():
                    try:
                        await client.send_text(error_json)
                    except Exception:
                        pass
                continue
                
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
            inference_time = 0.0

            if image_id == "raw_image" or image_id == "processed_image" or image_id == "pre_crop":
                # Bypass inference for base/raw images, simply send them directly
                overlay = img
            else:
                # Run inference in a background thread to prevent blocking the async loop
                import time
                start_time = time.time()
                score, threshold, overlay = await asyncio.to_thread(
                    handler.run_inference, image_id, img
                )
                inference_time = time.time() - start_time
                
            if overlay is not None:
                if image_id in ["processed_image", "raw_image", "pre_crop"]:
                    print(f"\n[DEBUG] Broadcasting {image_id} for Cam {camera_id} via WebSocket. Size: {overlay.shape}")
                # Broadcast the result
                await broadcast_result(camera_id, image_id, score, threshold, overlay, inference_time)

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
    
    # Start GPIO Monitoring Task
    gpio_monitor_task = asyncio.create_task(monitor_gpio_status())
    
    for cam_name, cam_cfg in config.get("tcp_receivers", {}).items():
        task = asyncio.create_task(
            start_tcp_server(cam_cfg["port"], cam_cfg["id"])
        )
        tcp_tasks.append(task)
        
    loop = asyncio.get_running_loop()
    
    mqtt_config = config.get("mqtt", {})
    mqtt_host = mqtt_config.get("broker", "localhost")
    mqtt_port = mqtt_config.get("port", 1883)
    mqtt_user = mqtt_config.get("username", None)
    mqtt_pass = mqtt_config.get("password", None)
    
    global mqtt_client
    try:
        mqtt_client = mqtt.Client(userdata={'loop': loop}, callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
    except AttributeError:
        mqtt_client = mqtt.Client(userdata={'loop': loop})
        
    if mqtt_user and mqtt_pass:
        mqtt_client.username_pw_set(mqtt_user, mqtt_pass)
        
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    
    try:
        mqtt_client.connect(mqtt_host, mqtt_port, 60)
        mqtt_client.loop_start()
        print(f"Started MQTT client connecting to {mqtt_host}:{mqtt_port}")
    except Exception as e:
        print(f"Could not connect to MQTT Broker: {e}")
        
    yield
    
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        
    # Clean up tasks on shutdown
    gpio_monitor_task.cancel()
    for task in tcp_tasks:
        task.cancel()

app = FastAPI(lifespan=lifespan)

# Mount static files (Dashboard HTML/JS/CSS)
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")
@app.get("/")
async def get_dashboard_hub():
    with open(f"{static_dir}/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/camera")
async def get_camera_view(id: int = 0):
    with open(f"{static_dir}/camera.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/mqtt-debug")
async def get_mqtt_debug_view():
    with open(f"{static_dir}/mqtt_debug.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/calibrate")
async def get_calibrate_view():
    with open(f"{static_dir}/calibrate.html", "r") as f:
        return HTMLResponse(content=f.read())

class CalibrateSaveRequest(BaseModel):
    camera_id: str
    original_width: int
    original_height: int
    image_base64: str
    regions: Dict

@app.post("/api/calibrate/save")
async def save_calibration(req: CalibrateSaveRequest):
    try:
        # Decode base64 image
        if ',' in req.image_base64:
            b64_data = req.image_base64.split(',')[1]
        else:
            b64_data = req.image_base64
            
        img_bytes = base64.b64decode(b64_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"status": "error", "message": "Failed to decode image"}
            
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(base_dir, "calibration_data")
        os.makedirs(config_dir, exist_ok=True)
        
        cam_dir = os.path.join(config_dir, f"cam_{req.camera_id}")
        os.makedirs(cam_dir, exist_ok=True)
        
        m1 = req.regions.get("m1")
        m2 = req.regions.get("m2")
        box_wall = req.regions.get("box_wall", [])
        rois = req.regions.get("rois", [])
        
        # Crop and save templates
        if m1:
            crop_m1 = img[m1["y"]:m1["y"]+m1["h"], m1["x"]:m1["x"]+m1["w"]]
            cv2.imwrite(os.path.join(cam_dir, "m1.jpg"), crop_m1)
        if m2:
            crop_m2 = img[m2["y"]:m2["y"]+m2["h"], m2["x"]:m2["x"]+m2["w"]]
            cv2.imwrite(os.path.join(cam_dir, "m2.jpg"), crop_m2)
            
        # Save config JSON
        config_data = {
            "camera_id": req.camera_id,
            "reference_size": {"width": req.original_width, "height": req.original_height},
            "marks": {"m1": m1, "m2": m2},
            "box_wall": box_wall,
            "rois": rois
        }
        
        with open(os.path.join(cam_dir, "pre_processing_config.json"), "w") as f:
            json.dump(config_data, f, indent=4)
            
        return {"status": "success"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

class ResetCalibrationRequest(BaseModel):
    camera_id: str

@app.post("/api/calibrate/reset")
async def reset_calibration(req: ResetCalibrationRequest):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(base_dir, "calibration_data")
        cam_dir = os.path.join(config_dir, f"cam_{req.camera_id}")
        config_path = os.path.join(cam_dir, "pre_processing_config.json")
        if os.path.exists(config_path):
            os.remove(config_path)
            # Also clean up M1/M2 templates
            for f in ["m1.jpg", "m2.jpg"]:
                p = os.path.join(cam_dir, f)
                if os.path.exists(p):
                    os.remove(p)
        return {"status": "success"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

class GlobalStatusConfig(BaseModel):

    settings: Dict[str, bool]

@app.get("/api/config/global-status")
async def get_global_status_config():
    settings = {}
    models_config = config.get("models", {})
    for k, v in models_config.items():
        if k == "pre_crop": continue
        settings[k] = v.get("use_for_global", True)
    return settings

@app.post("/api/config/global-status")
async def update_global_status_config(new_config: GlobalStatusConfig):
    models_config = config.get("models", {})
    for k, v in new_config.settings.items():
        if k in models_config:
            models_config[k]["use_for_global"] = v
            
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
        
    return {"status": "success"}

class SysCommand(BaseModel):
    target: str
    action: str

@app.post("/api/sys/command")
async def send_sys_command(cmd: SysCommand):
    global mqtt_client
    if not mqtt_client:
        return {"status": "error", "message": "MQTT client not initialized"}
        
    if cmd.action not in ["restart", "shutdown"]:
        return {"status": "error", "message": "Invalid action"}
        
    command_topic_template = config.get("mqtt", {}).get("command_topic", "{target}/sys/command")
    topic = command_topic_template.replace("{target}", cmd.target)
    payload = json.dumps({"system": cmd.action})
    
    try:
        mqtt_client.publish(topic, payload, retain=False)
        return {"status": "success", "action": cmd.action, "target": cmd.target}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/sys/devices")
async def get_sys_devices():
    # Fix: managed_devices is under mqtt block
    managed = config.get("mqtt", {}).get("managed_devices", [])
    if not managed: # Fallback to top-level if not under mqtt
        managed = config.get("managed_devices", [])
    
    response = []
    for device in managed:
        status_info = device_statuses.get(device, {"status": "offline", "last_seen": 0})
        response.append({
            "hostname": device,
            "status": status_info["status"],
            "last_seen": status_info["last_seen"]
        })
        
    return {"devices": response}

class GpioConfigUpdate(BaseModel):
    settings: Dict[str, bool]

@app.get("/api/config/gpio")
async def get_gpio_config():
    return config.get("gpio_triggers", {})

@app.post("/api/config/gpio")
async def update_gpio_config(new_config: GpioConfigUpdate):
    if "gpio_triggers" not in config:
        config["gpio_triggers"] = {}
        
    for k, v in new_config.settings.items():
        config["gpio_triggers"][str(k)] = v
        
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
        
    return {"status": "success"}

class DemoRequest(BaseModel):
    camera_id: int

@app.post("/api/demo/start")
async def start_demo_mode(req: DemoRequest):
    global demo_active, demo_tasks
    c_id = req.camera_id
    if not demo_active.get(c_id, False):
        demo_active[c_id] = True
        demo_tasks[c_id] = asyncio.create_task(demo_loop(c_id))
    return {"status": "started", "camera_id": c_id}

@app.post("/api/demo/stop")
async def stop_demo_mode(req: DemoRequest):
    global demo_active, demo_tasks
    c_id = req.camera_id
    if demo_active.get(c_id, False):
        demo_active[c_id] = False
        if c_id in demo_tasks and demo_tasks[c_id]:
            demo_tasks[c_id].cancel()
    return {"status": "stopped", "camera_id": c_id}

class GlobalDemoRequest(BaseModel):
    interval: float = 5.0

@app.post("/api/global-demo/start")
async def start_global_demo_mode(req: GlobalDemoRequest):
    global global_demo_active, global_demo_task, global_demo_interval, global_demo_paused
    global_demo_interval = max(1.0, float(req.interval)) # Ensure minimum 1 second
    global_demo_paused = False
    
    if not global_demo_active:
        global_demo_active = True
        if global_demo_task and not global_demo_task.done():
            global_demo_task.cancel()
        global_demo_task = asyncio.create_task(global_demo_loop())
    return {"status": "started", "interval": global_demo_interval}

@app.post("/api/global-demo/stop")
async def stop_global_demo_mode():
    global global_demo_active, global_demo_task, global_demo_paused
    if global_demo_active:
        global_demo_active = False
        global_demo_paused = False
        if global_demo_task and not global_demo_task.done():
            global_demo_task.cancel()
    return {"status": "stopped"}

class GlobalDemoUpdateIntervalRequest(BaseModel):
    interval: float

@app.post("/api/global-demo/update-interval")
async def update_global_demo_interval(req: GlobalDemoUpdateIntervalRequest):
    global global_demo_interval
    global_demo_interval = max(1.0, float(req.interval))
    return {"status": "updated", "interval": global_demo_interval}

@app.post("/api/global-demo/resume")
async def resume_global_demo_mode():
    global global_demo_paused
    global_demo_paused = False
    return {"status": "resumed"}

@app.get("/api/global-demo/status")
async def get_global_demo_status():
    global global_demo_active, global_demo_interval, global_demo_paused
    return {
        "active": global_demo_active,
        "interval": global_demo_interval,
        "paused": global_demo_paused
    }

class NGCropData(BaseModel):
    crop_id: str
    score: float

class NGReportPayload(BaseModel):
    crops: List[NGCropData]

@app.post("/api/stats/report-ng")
async def report_ng(payload: NGReportPayload):
    today = datetime.now().strftime("%Y-%m-%d")
    stats = load_ng_stats()
    
    logged_crops = []
    
    for crop in payload.crops:
        crop_id_str = str(crop.crop_id)
        if crop_id_str not in stats:
            stats[crop_id_str] = {}
            
        if today not in stats[crop_id_str]:
            stats[crop_id_str][today] = {"ng_count": 0, "total_score": 0.0, "avg_score": 0.0}
            
        day_stats = stats[crop_id_str][today]
        day_stats["ng_count"] += 1
        day_stats["total_score"] += crop.score
        day_stats["avg_score"] = day_stats["total_score"] / day_stats["ng_count"]
        
        logged_crops.append(crop_id_str)
        
    save_ng_stats(stats)
    return {"status": "logged", "crops": logged_crops, "date": today}

@app.get("/api/stats/ng-report")
async def get_ng_report(days: int = 1):
    stats = load_ng_stats()
    report = []
    
    # Active crops from config (models configured with use_for_global=True)
    # We allow reporting on all models that have an entry, but frontend handles filtering
    valid_crops = list(config.get("models", {}).keys())
    
    # Calculate cutoff date string for simple filtering
    from datetime import timedelta
    cutoff_date = (datetime.now() - timedelta(days=days-1)).strftime("%Y-%m-%d") if days > 0 else "0000-00-00"
    
    for crop_id, dates_data in stats.items():
        if crop_id not in valid_crops:
            continue
            
        total_ng_count = 0
        total_score_sum = 0.0
        
        for date_str, data in dates_data.items():
            if days <= 0 or date_str >= cutoff_date:
                total_ng_count += data.get("ng_count", 0)
                total_score_sum += data.get("total_score", 0.0)
                
        if total_ng_count > 0:
            avg_score = total_score_sum / total_ng_count
            report.append({
                "crop_id": crop_id,
                "ng_count": total_ng_count,
                "avg_score": avg_score
            })
            
    # Sort descending by NG count
    report.sort(key=lambda x: x["ng_count"], reverse=True)
    return {"report": report}

@app.post("/api/stats/ng-report/reset")
async def reset_ng_report():
    save_ng_stats({})
    return {"status": "reset_success"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    
    # Send initial GPIO state immediately upon connection
    try:
        init_msgs = [
            {"type": "gpio", "pin": 23, "state": gpio_states[23]["state"], "last_changed": gpio_states[23]["last_changed"]},
            {"type": "gpio", "pin": 24, "state": gpio_states[24]["state"], "last_changed": gpio_states[24]["last_changed"]}
        ]
        for msg in init_msgs:
            if msg["last_changed"] is not None:
                await websocket.send_text(json.dumps(msg))
    except Exception as e:
        print(f"Error sending init GPIO state: {e}")
        
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)

@app.websocket("/ws/mqtt-debug")
async def mqtt_debug_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    debug_clients[websocket] = set()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("action") == "subscribe":
                    topic = msg.get("topic")
                    if topic:
                        debug_clients[websocket].add(topic)
                        global mqtt_client
                        if mqtt_client:
                            mqtt_client.subscribe(topic)
                        await websocket.send_text(json.dumps({"topic": "sys", "payload": f"Subscribed to {topic}"}))
                elif msg.get("action") == "unsubscribe":
                    topic = msg.get("topic")
                    if topic and topic in debug_clients[websocket]:
                        debug_clients[websocket].remove(topic)
                        await websocket.send_text(json.dumps({"topic": "sys", "payload": f"Unsubscribed from {topic}"}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        if websocket in debug_clients:
            del debug_clients[websocket]

@app.post("/api/exit-kiosk")
async def exit_kiosk():
    try:
        subprocess.run(["killall", "chromium-browser"], check=False)
        subprocess.run(["killall", "chromium"], check=False)
        return {"status": "success", "message": "Kiosk mode exited"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    host = config.get("server", {}).get("host", "0.0.0.0") if config else "0.0.0.0"
    port = config.get("server", {}).get("port", 8000) if config else 8000
    uvicorn.run("main:app", host=host, port=port, reload=False)
