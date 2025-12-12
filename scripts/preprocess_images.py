import cv2
import numpy as np
import os
from pathlib import Path

INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/processed"

NORMALIZE_LIGHTING = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8,8)

ENHANCE_SHADOWS = True
GAMMA_VALUE = 1.2

ENHANCE_EDGES = True
EDGE_STRENGTH = 1.0

RESIZE_TO = None

class Preprocessor:
    def __init__(self):
        self.processed_count = 0
    
    def normalize_lighting(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
        l_clahe = clahe.apply(l)
        result = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)
        return result
    
    def enhance_shadows(self, img):
        invGamma = 1.0 / GAMMA_VALUE
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)
    
    def enhance_edges(self, img):
        blurred = cv2.GaussianBlur(img, (0, 0), 3.0)
        return cv2.addWeighted(img, 1.0 + EDGE_STRENGTH, blurred, -EDGE_STRENGTH, 0)
    
    def preprocess_image(self, img):
        result = img.copy()
        
        if NORMALIZE_LIGHTING:
            result = self.normalize_lighting(result)
        
        if ENHANCE_SHADOWS:
            result = self.enhance_shadows(result)
        
        if ENHANCE_EDGES:
            result = self.enhance_edges(result)
        
        if RESIZE_TO is not None:
            result = cv2.resize(result, RESIZE_TO)
        
        return result
    
    def process_directory(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        input_path = Path(INPUT_DIR)
        
        image_files = []
        for ext in {'.jpg', '.jpeg', '.png', '.bmp'}:
            image_files.extend(input_path.rglob(f'*{ext}'))
            image_files.extend(input_path.rglob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No images found in {INPUT_DIR}")
            return
        
        print(f"Processing {len(image_files)} images...")
        
        for i, img_path in enumerate(image_files, 1):
            try:
                img = cv2.imread(str(img_path))
                if img is None: continue
                
                processed = self.preprocess_image(img)
                
                relative_path = img_path.relative_to(input_path)
                output_path = Path(OUTPUT_DIR) / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                cv2.imwrite(str(output_path), processed)
                self.processed_count += 1
                
                if i % 10 == 0:
                    print(f"Progress: {i}/{len(image_files)}")
                    
            except Exception as e:
                print(f"Error {img_path.name}: {e}")
        
        print(f"Complete. {self.processed_count} images saved to {OUTPUT_DIR}")
        
        if len(image_files) > 0:
            self.show_comparison(image_files[0])
    
    def show_comparison(self, img_path):
        original = cv2.imread(str(img_path))
        if original is None: return
        
        processed = self.preprocess_image(original)
        
        h, w = original.shape[:2]
        if w > 800:
            scale = 800 / w
            new_dim = (800, int(h * scale))
            original = cv2.resize(original, new_dim)
            processed = cv2.resize(processed, new_dim)
            
        comparison = np.hstack([original, processed])
        
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(comparison, "Processed", (original.shape[1]+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        
        cv2.imshow('Preprocessing Check', comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    Preprocessor().process_directory()