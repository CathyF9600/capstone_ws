from ultralytics import YOLO
import cv2
import torch

def load_model(model_path):
    """Load the trained YOLOv8 model."""
    return YOLO(model_path)

def run_inference(model, image_path):
    """Run inference on a single image and return bounding box results."""
    results = model(image_path)
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # Convert bounding boxes to NumPy array
        confidences = r.boxes.conf.cpu().numpy()  # Get confidence scores
        class_ids = r.boxes.cls.cpu().numpy()  # Get class IDs
        
        return [{
            'bbox': box.tolist(),
            'confidence': float(conf),
            'class_id': int(cls_id)
        } for box, conf, cls_id in zip(boxes, confidences, class_ids)]

if __name__ == "__main__":
    model_path = "./best.pt"  
    image_path = "./farm_image.jpeg"  
    
    model = load_model(model_path)
    results = run_inference(model, image_path)
    
    print("Bounding Box Results:", results)

