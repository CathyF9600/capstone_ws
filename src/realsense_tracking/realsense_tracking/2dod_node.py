import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import torch
from ultralytics import YOLO
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy, qos_profile_system_default

class YOLOv8ROS2(Node):
    def __init__(self):
        super().__init__('realsense_tracking')
        # Load the YOLOv8 model
        self.model = YOLO('./best.pt')
        
        # Create a subscriber to the /2dod/img topic
        self.subscription = self.create_subscription(
            Image,
            '/vision_2d/img',
            self.image_callback,
            qos_profile_system_default  # QoS profile depth
        )
        # self.subscription  # prevent unused variable warning
        # ROS 2 Publisher for bounding boxes
        self.bbox_publisher = self.create_publisher(
            Detection2DArray,
            '/vision_2d/bbx',
            qos_profile_system_default
        )
        # Bridge for converting ROS2 Image messages to OpenCV format
        self.bridge = CvBridge()

    def image_callback(self, msg):
        self.get_logger().info('Received image')

        # Convert ROS2 Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Run inference
        results = self.run_inference(cv_image)

        # Publish bounding boxes
        self.publish_bboxes(results)


    def run_inference(self, image):
        """Run inference on an OpenCV image and return bounding box results."""
        results = self.model(image)
        detections = []

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()  # Bounding boxes
            confidences = r.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = r.boxes.cls.cpu().numpy()  # Class IDs

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                detections.append({
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'class_id': int(cls_id)
                })

        return detections


    def publish_bboxes(self, results):
        """Publish detected bounding boxes as a Detection2DArray message."""
        detection_array_msg = Detection2DArray()

        for result in results:
            detection = Detection2D()
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = result['class_id']
            hypothesis.score = result['confidence']

            # Populate bounding box coordinates
            x_min, y_min, x_max, y_max = result['bbox']
            detection.bbox.center.position.x = (x_min + x_max) / 2.0
            detection.bbox.center.position.y = (y_min + y_max) / 2.0
            detection.bbox.size_x = x_max - x_min
            detection.bbox.size_y = y_max - y_min

            detection.results.append(hypothesis)
            detection_array_msg.detections.append(detection)

        self.bbox_publisher.publish(detection_array_msg)
        self.get_logger().info(f'Published {len(results)} bounding boxes to /2dod/bbx')

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv8ROS2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()