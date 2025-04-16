#!/usr/bin/env python3

import cv2
from ultralytics import YOLO
import torch
import serial
import threading
import time

### ---------- Arduino Interface ----------

class ArduinoInterface:
    def __init__(self, port='/dev/ttyACM0', baudrate=9600):
        self.ser = None
        try:
            self.ser = serial.Serial(port, baudrate, timeout=0.1)
            if not self.ser.is_open:
                raise serial.SerialException("Port exists but is not open.")
            print("Arduino connected successfully on port {}".format(port))
        except (serial.SerialException, OSError) as e:
            print("Error: Could not connect to Arduino on port {}. {}".format(port, e))
            self.ser = None

        self.lock = threading.Lock()
        self.waiting_for_confirm = False
        self.last_command = None

        if self.ser:
            self.confirm_thread = threading.Thread(target=self._listen_for_confirmations)
            self.confirm_thread.daemon = True
            self.confirm_thread.start()

    def _listen_for_confirmations(self):
        last_check_time = time.time()
        TIMEOUT = 3.0  # seconds

        while self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode(errors="ignore").strip()
                    print(f"[Arduino → Jetson] Received: '{line}'")

                    expected = f"{self.last_command}_CONFIRM" if self.last_command else None
                    if expected and line.upper() == expected.upper():
                        with self.lock:
                            self.waiting_for_confirm = False
                            self.last_command = None
                            last_check_time = time.time()  # Reset timer
                    else:
                        print(f"Unexpected response: '{line}' (expected: '{expected}')")
                else:
                    # Check timeout
                    if self.waiting_for_confirm and (time.time() - last_check_time > TIMEOUT):
                        with self.lock:
                            print("Timeout waiting for confirmation. Resetting state.")
                            self.waiting_for_confirm = False
                            self.last_command = None
                            last_check_time = time.time()
            except serial.SerialException:
                print("Error: Lost connection to Arduino.")
                self.ser = None
                break
            time.sleep(0.05)

    def send_command(self, command):
        if not self.is_connected():
            print("Error: Arduino is not connected.")
            return False

        should_wait_for_confirm = command.upper() in ["ON", "OFF"]

        with self.lock:
            if should_wait_for_confirm and self.waiting_for_confirm:
                print("Waiting for confirmation before sending another command.")
                return False
            self.last_command = command if should_wait_for_confirm else None
            self.waiting_for_confirm = should_wait_for_confirm

        full_command = "{}\n".format(command)
        try:
            self.ser.write(full_command.encode())
            print("[Jetson to Arduino] Sent command: {}".format(command))
            return True
        except serial.SerialException:
            print("Error: Failed to send command.")
            self.ser = None
            return False

    def is_connected(self):
        return self.ser is not None and self.ser.is_open

    def close(self):
        if self.ser:
            self.ser.close()
            print("Serial connection closed.")
            self.ser = None

### ---------- YOLO and Video ----------

def gstreamer_pipeline(capture_width=320, capture_height=240, display_width=320, display_height=240, framerate=15, flip_method=0):
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
    return YOLO(model_path)

def is_centered(box, frame_shape, tolerance_ratio=0.1):
    x1, y1, x2, y2 = box
    box_center_x = (x1 + x2) / 2
    frame_center_x = frame_shape[1] / 2
    tolerance = tolerance_ratio * frame_shape[1]
    return abs(box_center_x - frame_center_x) < tolerance

def draw_boxes_and_check_center(frame, results, model):
    detected_and_centered = False
    best_detection = None
    highest_conf = 0.0

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
            """if is_centered((x1, y1, x2, y2), frame.shape):
                detected_and_centered = True"""
            if conf > highest_conf and is_centered((x1, y1, x2, y2), frame.shape):
                highest_conf = conf
                best_detection = (box, conf, cls_id)

    return frame, best_detection

    """if detected_and_centered:
        arduino.send_command("OPEN")
    else:
        arduino.send_command("CLOSE")

    return frame"""

def show_camera_with_detection(model, arduino):
    window_title = "YOLOv11n CSI Camera"
    video_capture = cv2.VideoCapture(
        gstreamer_pipeline(capture_width=320, capture_height=240, display_width=320, display_height=240, framerate=15, flip_method=2),
        cv2.CAP_GSTREAMER
    )

    gripper_state = "OPEN"
    arduino.send_command("OFF")
    last_centered_time = time.time()
    last_action_time = time.time()


    if video_capture.isOpened():
        try:
            cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            frame_count = 0
            inference_interval = 2

            while True:
                ret_val, frame = video_capture.read()
                if not ret_val:
                    print("Failed to read frame")
                    break

                if frame_count % inference_interval == 0:
                    results = model.predict(frame, verbose=False)
                    frame, best_detection = draw_boxes_and_check_center(frame, results, model)

                    current_time = time.time()

                    if best_detection and best_detection[1] > 0.7:
                        if gripper_state == "OPEN" and (current_time - last_centered_time > 2):
                            arduino.send_command("ON")
                            gripper_state = "CLOSE"
                            last_action_time = current_time
                        last_centered_time = current_time
                    else:
                        if gripper_state == "CLOSE" and (current_time - last_centered_time > 2):
                            arduino.send_command("OFF")
                            gripper_state = "OPEN"
                            last_action_time = current_time


                frame_count += 1

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
            arduino.close()
    else:
        print("Error: Unable to open camera")

### ---------- Main ----------

if __name__ == "__main__":
    model_path = "src/scripts/best_lime_v11.pt"  # Update path as needed
    model = load_model(model_path)
    arduino = ArduinoInterface(port='/dev/ttyACM0', baudrate=9600)
    time.sleep(1)
    show_camera_with_detection(model, arduino)
