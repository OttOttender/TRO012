import cv2
import os
from datetime import datetime

DROIDCAM_IP = "192.168.3.19"
DROIDCAM_PORT = "4747"
CAMERA_SOURCE = f"http://{DROIDCAM_IP}:{DROIDCAM_PORT}/video"
OUTPUT_DIR = "data/raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print(f"Connecting to: {CAMERA_SOURCE}")
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Connection failed. Check IP and DroidCam app.")
        return

    capture_count = len(os.listdir(OUTPUT_DIR))
    print(f"✅ Connected. Press 'C' to Capture, 'Q' to Quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost stream...")
            break

        display = frame.copy()
        h, w = display.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.line(display, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
        cv2.line(display, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)
        cv2.putText(display, f"Captured: {capture_count}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "[C] Capture  [Q] Quit", (10, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow('Data Collector', display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
      
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(OUTPUT_DIR, f"battery_{timestamp}.jpg")
            cv2.imwrite(filename, frame) # Save clean frame (no crosshair)
            
            capture_count += 1
            print(f"Saved: {filename}")

       
            flash = display.copy()
            flash[:] = (255, 255, 255)
            cv2.imshow('Data Collector', flash)
            cv2.waitKey(50)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()