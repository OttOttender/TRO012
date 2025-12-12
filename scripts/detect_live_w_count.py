import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

DROIDCAM_IP = "192.168.3.19"   
DROIDCAM_PORT = "4747"
CAMERA_SOURCE = f"http://{DROIDCAM_IP}:{DROIDCAM_PORT}/video"
MODEL_PATH = r"models/battery_detector/weights/best.pt"

CONFIDENCE_THRESHOLD = 0.4  
IOU_THRESHOLD = 0.6         

APPLY_PREPROCESSING = True
CLAHE_CLIP_LIMIT = 2.0   
CLAHE_GRID_SIZE = (8,8)
GAMMA_VALUE = 1.2        
EDGE_STRENGTH = 1.0      

class BatteryCounter:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = YOLO(model_path)
        print(f"Model loaded. Classes: {self.model.names}")
        
        self.fps_time = time.time()
        self.fps = 0
        np.random.seed(42) 
        self.colors = np.random.uniform(0, 255, size=(100, 3))
        
    def preprocess_frame(self, img):
        if not APPLY_PREPROCESSING: return img
        result = img.copy()
        
        # CLAHE
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
        l_clahe = clahe.apply(l)
        result = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)
        
        # Gamma Correction
        invGamma = 1.0 / GAMMA_VALUE
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        result = cv2.LUT(result, table)
        
        # Sharpening
        blurred = cv2.GaussianBlur(result, (0, 0), 3.0)
        result = cv2.addWeighted(result, 1.0 + EDGE_STRENGTH, blurred, -EDGE_STRENGTH, 0)
        
        return result

    def draw_results(self, frame, results):
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        counts = {}
        total_batteries = 0

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()

            for box, track_id, conf, cls_id in zip(boxes, track_ids, confs, clss):
                x1, y1, x2, y2 = map(int, box)
                cls_id = int(cls_id)
                name = self.model.names[cls_id]
                
                counts[name] = counts.get(name, 0) + 1
                total_batteries += 1
                
                color = self.colors[cls_id]
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"#{int(track_id)} {name} {conf:.2f}"
                (w_text, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(display_frame, (x1, y1 - 20), (x1 + w_text, y1), color, -1)
                cv2.putText(display_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        panel_w = 320
        cv2.rectangle(display_frame, (w - panel_w, 0), (w, h), (30, 30, 30), -1)
        cv2.putText(display_frame, "BATTERY COUNTER", (w - panel_w + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"TOTAL: {total_batteries}", (w - panel_w + 10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        y = 130
        if not counts:
             cv2.putText(display_frame, "Scanning...", (w - panel_w + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        else:
            for name, count in sorted(counts.items()):
                try:
                    c_id = list(self.model.names.values()).index(name)
                    col = self.colors[c_id]
                except: col = (255,255,255)
                cv2.putText(display_frame, f"{name}: {count}", (w - panel_w + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
                y += 35

        cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (w - panel_w + 10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return display_frame

    def run(self):
        print(f"Connecting to: {CAMERA_SOURCE}")
        cap = cv2.VideoCapture(CAMERA_SOURCE)
        
        if not cap.isOpened():
            print("Connection failed.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                cap = cv2.VideoCapture(CAMERA_SOURCE)
                continue
            
            curr_time = time.time()
            self.fps = 1 / (curr_time - self.fps_time) if (curr_time - self.fps_time) > 0 else 0
            self.fps_time = curr_time

            ai_frame = self.preprocess_frame(frame)
            results = self.model.track(
                ai_frame, 
                persist=True, 
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                agnostic_nms=True,
                verbose=False
            )
            
            final_frame = self.draw_results(frame, results)
            cv2.imshow('Battery Counter', final_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    BatteryCounter(MODEL_PATH).run()