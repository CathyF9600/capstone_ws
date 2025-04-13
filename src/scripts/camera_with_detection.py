#!/usr/bin/env python3

# MIT License
# Copyright (c) 2019-2022 JetsonHacks

import cv2
from ultralytics import YOLO
import torch

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )

def load_model(model_path):
    """Load the trained YOLOv8 model."""
    return YOLO(model_path)

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

def blur_frame_like_low_res(frame, downscale_factor=4):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // downscale_factor, h // downscale_factor), interpolation=cv2.INTER_LINEAR)
    blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return blurred


def show_camera_with_detection(model):
    window_title = "YOLOv8 CSI Camera"
    # video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
    video_capture = cv2.VideoCapture(
        gstreamer_pipeline(capture_width=640, capture_height=360, display_width=640, display_height=360, flip_method=2),
        cv2.CAP_GSTREAMER
    )
    if video_capture.isOpened():
        try:
            cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                frame = blur_frame_like_low_res(frame, downscale_factor=4)

                # if not ret_val:
                #     print("Failed to read frame")
                #     break

                # Run YOLOv8 inference
                results = model(frame)
                # Draw bounding boxes
                frame = draw_boxes(frame, results)

                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, frame)
                else:
                    break

                keyCode = cv2.waitKey(1) & 0xFF
                if keyCode == 27 or keyCode == ord('q'):
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


def show_camera():
    window_title = "CSI Camera"

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                frame = downscale_frame(frame)

                print('frame', type(frame))
                # Check to see if the user closed the window
                # Under GTK (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, frame)
                else:
                    break 
                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")

if __name__ == "__main__":
    model_path = "src/scripts/best.pt"  # Update with your actual model path
    # show_camera()
    model = load_model(model_path)
    show_camera_with_detection(model)

