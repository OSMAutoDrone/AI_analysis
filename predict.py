from ultralytics import YOLO
import cv2

model_path = 'runs/segment/train21/weights/best.pt'
base_path = r"C:/Users/ti-da/OneDrive/Documents/PMC/AI_analysis/data/"
image_path = f"{base_path}testGeoImage02.tif"

model = YOLO(model_path)
results = model.predict(image_path, save_conf=True, show_boxes=False, save=True)