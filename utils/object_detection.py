from ultralytics import YOLO
import os

#THE FUNCTION TOP FIND OBJECTS IN THE FRAME
def detect_objects(frame, model_path = "models/yolov8n.pt"):
    model = YOLO(model_path)
    results = model(frame)
    detections = []
    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(cls)]
            detections.append({"box": (x1, y1, x2, y2), "class": class_name, "confidence": float(conf)})
    return detections