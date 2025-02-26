import cv2
import torch
from ultralytics import YOLO

#USES CUDA DEVICE IF AVAILABE
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

#MODEL LOAD
model = YOLO("yolov8s.pt").to(device)

#CAMERA OPEN 1
cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    #RUN YOLO
    results = model(frame)

    #SHOW THE RESULTS ON THE FRAME
    annotated_frame = results[0].plot()

    #SHOW THE FRAME
    cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    #QUIT ON Q PRESS
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")