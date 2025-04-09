import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import os
from message_filters import ApproximateTimeSynchronizer, Subscriber
from datetime import datetime

class RGBDRecorder(Node):
    def __init__(self):
        super().__init__('rgbd_recorder')
        self.bridge = CvBridge()
        self.last_saved_time = self.get_clock().now()

        # Subscribers with time sync
        self.rgb_sub = Subscriber(self, Image, '/rgbd_image')
        self.depth_sub = Subscriber(self, Image, '/depth_image')

        self.sync = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.callback)

        self.output_dir = 'rgbd_npy'
        os.makedirs(self.output_dir, exist_ok=True)

    def callback(self, rgb_msg, depth_msg):
        now = self.get_clock().now()
        time_since_last = (now - self.last_saved_time).nanoseconds / 1e9

        if time_since_last < 1.0:
            return  # Only save every 1 second

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')  # HxWx3
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')  # HxW

            if depth_image.ndim == 3:
                depth_image = depth_image[:, :, 0]  # If it's 3D, collapse to 2D

            h, w = depth_image.shape
            rgb_resized = cv2.resize(rgb_image, (w, h))

            rgbd = np.dstack((rgb_resized, depth_image))  # HxWx4

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f'rgbd_{timestamp}.npy')
            np.save(filename, rgbd)
            self.get_logger().info(f"Saved: {filename}")
            self.last_saved_time = now

        except Exception as e:
            self.get_logger().error(f"Error in callback: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = RGBDRecorder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
