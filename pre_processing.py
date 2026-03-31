import asyncio
import cv2
import numpy as np
import json
import struct
import base64
import os
import glob

# Constants
PROXY_PORTS = {
    4: 8084, # listends on 8084 (cameras connect here)
    5: 8085, 
    6: 8086
}
MAIN_PORTS = {
    4: 9084, # forwards to main.py listening on 9084
    5: 9085,
    6: 9086
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(BASE_DIR, "calibration_data")

# Global Cache for configurations and templates
cam_configs = {}

def load_camera_config(cam_id):
    """Loads JSON config and template images for a camera"""
    config_path = os.path.join(CALIB_DIR, f"cam_{cam_id}", "pre_processing_config.json")
    if not os.path.exists(config_path):
        return None
        
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
            
        # Load templates
        m1_path = os.path.join(CALIB_DIR, f"cam_{cam_id}", "m1.jpg")
        m2_path = os.path.join(CALIB_DIR, f"cam_{cam_id}", "m2.jpg")
        
        m1_img = cv2.imread(m1_path, cv2.IMREAD_GRAYSCALE)
        m2_img = cv2.imread(m2_path, cv2.IMREAD_GRAYSCALE)
        
        cfg['loaded_templates'] = {'m1': m1_img, 'm2': m2_img}
        return cfg
    except Exception as e:
        print(f"Error loading config for cam {cam_id}: {e}")
        return None

def process_frame(img_bgr, cfg):
    """Executes the pre-processing pipeline"""
    if cfg is None or cfg.get('loaded_templates', {}).get('m1') is None:
        return {'raw_image': img_bgr} # Pass through if not configured
        
    # Resize raw image to match calibration reference size if specified
    ref_w = cfg.get('reference_size', {}).get('width', 0)
    ref_h = cfg.get('reference_size', {}).get('height', 0)
    if ref_w > 0 and ref_h > 0 and (img_bgr.shape[1] != ref_w or img_bgr.shape[0] != ref_h):
        img_bgr = cv2.resize(img_bgr, (ref_w, ref_h))
        
    # 1. Multi-Mark Image Alignment (Registration & Warping)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    m1_tpl = cfg['loaded_templates']['m1']
    m2_tpl = cfg['loaded_templates']['m2']
    m1_ref = cfg['marks']['m1']
    m2_ref = cfg['marks']['m2']
    
    if gray.shape[0] < m1_tpl.shape[0] or gray.shape[1] < m1_tpl.shape[1]:
        return {'raw_image': img_bgr}
        
    # Match M1 globally
    res1 = cv2.matchTemplate(gray, m1_tpl, cv2.TM_CCOEFF_NORMED)
    _, max_val1, _, max_loc1 = cv2.minMaxLoc(res1)
    
    # Expected relative offset for M2 based on M1
    expected_m2_dx = m2_ref['x'] - m1_ref['x']
    expected_m2_dy = m2_ref['y'] - m1_ref['y']
    
    # Search for M2 in a neighborhood
    search_x = max(0, max_loc1[0] + expected_m2_dx - 200)
    search_y = max(0, max_loc1[1] + expected_m2_dy - 200)
    search_w = min(gray.shape[1] - search_x, m2_tpl.shape[1] + 400)
    search_h = min(gray.shape[0] - search_y, m2_tpl.shape[0] + 400)
    
    roi_search2 = gray[search_y:search_y+search_h, search_x:search_x+search_w]
    if roi_search2.shape[0] < m2_tpl.shape[0] or roi_search2.shape[1] < m2_tpl.shape[1]:
        max_val2 = 0
        max_loc2 = (0, 0)
        m2_found_x = search_x
        m2_found_y = search_y
    else:
        res2 = cv2.matchTemplate(roi_search2, m2_tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val2, _, max_loc2 = cv2.minMaxLoc(res2)
        m2_found_x = search_x + max_loc2[0]
        m2_found_y = search_y + max_loc2[1]
    
    
    
    # Calculate transform matrix (Estimate Affine 2D)
    matrix = None
    if max_val1 > 0.55 and max_val2 > 0.55:
        pts_src = np.array([[max_loc1[0], max_loc1[1]], [m2_found_x, m2_found_y]], dtype=np.float32)
        pts_dst = np.array([[m1_ref['x'], m1_ref['y']], [m2_ref['x'], m2_ref['y']]], dtype=np.float32)
        
        # Prevent extreme rotation if points are too close together
        dist_src = np.linalg.norm(pts_src[0] - pts_src[1])
        if dist_src > 50:
            matrix, _ = cv2.estimateAffinePartial2D(pts_src, pts_dst)
    
    if matrix is not None:
        h, w = gray.shape
        aligned = cv2.warpAffine(img_bgr, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        aligned = img_bgr.copy() # fallback
        
    # 2. Shadow Removal (Disabled due to causing white images)
    # blurred = cv2.GaussianBlur(aligned, (0, 0), sigmaX=50, sigmaY=50)
    # blurred[blurred == 0] = 1 
    # shadow_removed = cv2.divide(aligned, blurred, scale=255)
    shadow_removed = aligned.copy()
    
    # 3. Pre-Crop or Box Wall Warp
    M_perspective = None
    if 'box_wall' in cfg and len(cfg['box_wall']) == 4:
        pts = cfg['box_wall']
        rect = np.zeros((4, 2), dtype="float32")
        pts_arr = np.array([[p['x'], p['y']] for p in pts])
        s = pts_arr.sum(axis=1)
        rect[0] = pts_arr[np.argmin(s)]
        rect[2] = pts_arr[np.argmax(s)]
        diff = np.diff(pts_arr, axis=1)
        rect[1] = pts_arr[np.argmin(diff)]
        rect[3] = pts_arr[np.argmax(diff)]
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        M_perspective = cv2.getPerspectiveTransform(rect, dst)
        color_box_wall = cv2.warpPerspective(aligned, M_perspective, (maxWidth, maxHeight))
        pre_crop = cv2.warpPerspective(shadow_removed, M_perspective, (maxWidth, maxHeight))
        left = 0
        top = 0
    else:
        # 3. Pre-Crop (Margin Cropping) - default margins if not in config
        top, bottom, left, right = 35, 40, 40, 35
        h, w = shadow_removed.shape[:2]
        color_box_wall = aligned[top:h-bottom, left:w-right]
        pre_crop = shadow_removed[top:h-bottom, left:w-right]
    
    # 4. Grayscale & Enhancement
    gray_pre_crop = cv2.cvtColor(pre_crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray_pre_crop)
    final_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    # 5. ROI Box Cropping and Masking
    bundles = {'processed_image': color_box_wall, 'raw_image': img_bgr}
    masked_surface = final_bgr.copy()
    
    for roi in cfg.get('rois', []):
        rx, ry, rw, rh = roi['x'], roi['y'], roi['w'], roi['h']
        if M_perspective is not None:
            roi_pts = np.array([
                [[rx, ry]],
                [[rx + rw, ry]],
                [[rx + rw, ry + rh]],
                [[rx, ry + rh]]
            ], dtype='float32')
            warped_roi_pts = cv2.perspectiveTransform(roi_pts, M_perspective)
            wx, wy, ww, wh = cv2.boundingRect(warped_roi_pts)
            wx, wy = max(0, wx), max(0, wy)
            ww = max(1, min(ww, final_bgr.shape[1] - wx))
            wh = max(1, min(wh, final_bgr.shape[0] - wy))
            crop_img = final_bgr[wy:wy+wh, wx:wx+ww]
            cv2.fillPoly(masked_surface, [warped_roi_pts.astype(int)], (0, 0, 0))
            bundles[roi['id']] = crop_img
        else:
            adj_x = max(0, rx - left)
            adj_y = max(0, ry - top)
            rw = max(1, min(rw, final_bgr.shape[1] - adj_x))
            rh = max(1, min(rh, final_bgr.shape[0] - adj_y))
            crop_img = final_bgr[adj_y:adj_y+rh, adj_x:adj_x+rw]
            bundles[roi['id']] = crop_img
            cv2.rectangle(masked_surface, (adj_x, adj_y), (adj_x+rw, adj_y+rh), (0, 0, 0), -1)
        
    bundles['masked_surface'] = masked_surface
    bundles['pre_crop'] = pre_crop
    return bundles

async def handle_proxy_client(reader, writer, camera_id):
    """Receives raw frames from wfzero, processes, and targets main.py"""
    addr = writer.get_extra_info('peername')
    print(f"[Proxy Cam {camera_id}] Connected: {addr}")
    
    cfg_path = os.path.join(CALIB_DIR, f"cam_{camera_id}", "pre_processing_config.json")
    last_mtime = 0
    if os.path.exists(cfg_path):
        last_mtime = os.path.getmtime(cfg_path)
    cfg = load_camera_config(camera_id)
    target_port = MAIN_PORTS.get(camera_id)
    
    if not target_port:
        print(f"Cam {camera_id} has no target port configured.")
        writer.close()
        return

    out_writer = None
    try:
        out_reader, out_writer = await asyncio.open_connection('127.0.0.1', target_port)
        print(f"[Proxy Cam {camera_id}] Connected to main.py at port {target_port}")
        
        while True:
            # Hot-reload config if changed
            try:
                if os.path.exists(cfg_path):
                    mtime = os.path.getmtime(cfg_path)
                    if mtime > last_mtime:
                        cfg = load_camera_config(camera_id)
                        last_mtime = mtime
                        print(f"[Proxy Cam {camera_id}] Hot-reloaded new config!")
                else:
                    if last_mtime != 0:
                        cfg = None
                        last_mtime = 0
                        print(f"[Proxy Cam {camera_id}] Config deleted. Tracking reset.")
            except Exception:
                pass

            # 1. Header
            try:
                header_data = await reader.readexactly(4)
            except asyncio.IncompleteReadError:
                break
            json_size = struct.unpack('>L', header_data)[0]
            
            # 2. JSON
            try:
                json_bytes = await reader.readexactly(json_size)
            except asyncio.IncompleteReadError:
                break
            metadata = json.loads(json_bytes.decode('utf-8'))
            image_id = metadata.get("id", "unknown")
            image_size = metadata.get("size", 0)
            
            # 3. JPEG
            if image_size <= 0: continue
            try:
                jpeg_data = await reader.readexactly(image_size)
            except asyncio.IncompleteReadError:
                break
                
            np_arr = np.frombuffer(jpeg_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None: continue
            
            # Run Process
            try:
                bundles = await asyncio.to_thread(process_frame, img, cfg)
            except Exception as e:
                # If the AI visual processing crashes, send the exact error traceback!
                bundles = {'error': str(e)}
            
            # If pass-through (no ROIs and no processed_image meaning it was uncalibrated)
            if 'raw_image' in bundles and len(bundles) == 1:
                bundles = {image_id: bundles['raw_image']}
                
            # Forward continuously
            for name, out_item in bundles.items():
                if name == 'error':
                    # Send an error payload instead of an image
                    error_payload = json.dumps({"id": "error", "message": out_item, "size": 0}).encode('utf-8')
                    out_header = struct.pack('>L', len(error_payload))
                    out_writer.write(out_header)
                    out_writer.write(error_payload)
                    await out_writer.drain()
                    continue
                    
                if out_item is None or getattr(out_item, 'size', 0) == 0:
                    continue
                    
                # Fix for cv2.error causing broken pipes to the camera
                try:
                    _, buffer = cv2.imencode('.jpg', out_item, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    img_bytes = buffer.tobytes()
                    
                    out_metadata = json.dumps({"id": name, "size": len(img_bytes)}).encode('utf-8')
                    out_header = struct.pack('>L', len(out_metadata))
                    
                    out_writer.write(out_header)
                    out_writer.write(out_metadata)
                    out_writer.write(img_bytes)
                    await out_writer.drain()
                except Exception as encoding_err:
                    print(f"[Cam {camera_id}] Error encoding/sending {name}: {encoding_err}")
            
    except ConnectionRefusedError:
        print(f"Cam {camera_id}: main.py not listening on {target_port}")
    except Exception as e:
        print(f"Proxy stream error cam {camera_id}: {e}")
    finally:
        writer.close()
        await writer.wait_closed()
        if out_writer:
            out_writer.close()
            await out_writer.wait_closed()
            print(f"[Proxy Cam {camera_id}] Disconnected from main.py")
        
async def proxy_server(port, cam_id):
    server = await asyncio.start_server(
        lambda r, w: handle_proxy_client(r, w, cam_id),
        host='0.0.0.0', port=port
    )
    print(f"Proxy for Cam {cam_id} listening on {port} (targeting {MAIN_PORTS[cam_id]})...")
    async with server:
        await server.serve_forever()

async def main():
    tasks = []
    for cam_id, port in PROXY_PORTS.items():
        tasks.append(asyncio.create_task(proxy_server(port, cam_id)))
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
