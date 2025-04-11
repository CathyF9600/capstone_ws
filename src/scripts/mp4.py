import numpy as np
import cv2
from ultralytics import YOLO
import torch

model_path = "src/scripts/best.pt"  # Update with your actual model path


def load_model(model_path):
    """Load the trained YOLOv8 model."""
    return YOLO(model_path)

model = load_model(model_path)

def draw_boxes(frame, results):
    """Draw bounding boxes and labels on the frame."""
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy()

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls_id)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# cap = cv2.VideoCapture('../../Downloads/farm_lime4.mp4')
cap = cv2.VideoCapture('src/scripts/farm_lime3.mp4')

while(cap.isOpened()):

    ret, frame = cap.read()

    if ret==True:
        # Run YOLOv8 inference
        results = model(frame)
        # Draw bounding boxes
        frame = draw_boxes(frame, results)
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()