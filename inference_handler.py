import sys
import os
import cv2
import numpy as np
import threading
import yaml

# Import Hailo Optimized Class
current_dir = os.path.dirname(os.path.abspath(__file__))
ai_receiver_deploy_path_internal = os.path.join(current_dir, "ai_receiver_deploy")
ai_receiver_deploy_path_external = os.path.join(os.path.dirname(current_dir), "ai_receiver_deploy")

if os.path.exists(ai_receiver_deploy_path_internal):
    ai_receiver_deploy_path = ai_receiver_deploy_path_internal
elif os.path.exists(ai_receiver_deploy_path_external):
    ai_receiver_deploy_path = ai_receiver_deploy_path_external
else:
    print(f"[Error] Could not find 'ai_receiver_deploy' folder.")
    print(f"Looked in: {ai_receiver_deploy_path_internal} and {ai_receiver_deploy_path_external}")
    sys.exit(1)

sys.path.append(ai_receiver_deploy_path)

try:
    from inference_hailo_rpi_optimized import HailoPatchCoreOptimized
except ModuleNotFoundError:
    print(f"[Error] Could not find inference_hailo_rpi_optimized.py in {ai_receiver_deploy_path}")
    print("Please ensure the 'ai_receiver_deploy' folder is located next to 'ai_dashboard' or inside it.")
    sys.exit(1)

class InferenceHandler:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.models = {}
        self.thresholds = {}
        self.infer_lock = threading.Lock()
        
        self.load_models()
        
    def load_models(self):
        models_config = self.config.get('models', {})
        for image_id, cfg in models_config.items():
            hef_path = cfg.get("hef", "")
            size = cfg.get("size", 224)
            threshold = float(cfg.get("threshold", 0.5))
            
            self.thresholds[image_id] = threshold
            if not hef_path or not os.path.exists(hef_path):
                print(f"[Warning] {image_id}: Model file missing or path empty ('{hef_path}').")
                continue
                
            try:
                self.models[image_id] = HailoPatchCoreOptimized(hef_path, size=size)
                print(f"[Inference] Loaded {image_id} | threshold={threshold}")
            except Exception as e:
                print(f"[Error] Failed to load {image_id}: {e}")

    def run_inference(self, image_id, img):
        """
        Runs Hailo inference on an image slice.
        Returns: Tuple of (score: float, threshold: float, overlay_image_bgr: np.ndarray)
        """
        model = self.models.get(image_id)
        threshold = self.thresholds.get(image_id, 0.5)
        
        if model is None:
            # Model not loaded, return early without drawing overlay
            return None, threshold, None

        with self.infer_lock:
            try:
                score, amap = model.infer(img)
            except Exception as e:
                print(f"[Inference] Error on {image_id}: {e}")
                return None, threshold, None
                
        # Heatmap Post-Processing and Overlay
        heatmap_resized = cv2.resize(amap, (img.shape[1], img.shape[0]))
        heatmap_vis = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-10)
        heatmap_uint8 = (heatmap_vis * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        overlay = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)
        
        return float(score), float(threshold), overlay
