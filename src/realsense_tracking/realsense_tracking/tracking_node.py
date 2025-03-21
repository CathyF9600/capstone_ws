#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import cv2
import numpy as np
from cv_bridge import CvBridge

class T265Tracker(Node):
    def __init__(self):
        super().__init__('realsense_tracking')
        self.bridge = CvBridge()

        # Subscribe to fisheye cameras
        self.create_subscription(Image, '/camera/fisheye1/image_raw', self.left_callback, 10)
        self.create_subscription(Image, '/camera/fisheye2/image_raw', self.right_callback, 10)

        # Subscribe to odometry for tracking
        self.create_subscription(Odometry, '/camera/odom/sample', self.odom_callback, 10)

    def left_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        cv2.imshow("Left Fisheye", img)
        cv2.waitKey(1)

    def right_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        cv2.imshow("Right Fisheye", img)
        cv2.waitKey(1)

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        self.get_logger().info(f"Position: x={pos.x:.2f}, y={pos.y:.2f}, z={pos.z:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = T265Tracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
