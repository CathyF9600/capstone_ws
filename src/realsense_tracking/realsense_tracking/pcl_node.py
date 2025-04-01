#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
import image_geometry
from rclpy.qos import qos_profile_system_default
import sensor_msgs.point_cloud2 as pc2

class PillarDetection(Node):
    def __init__(self):
        super().__init__('realsense_tracking')
        self.bridge = CvBridge()
        self.camera_model = None
        
        self.create_subscription(CameraInfo, '/camera/fisheye1/camera_info', self.camera_info_callback, qos_profile_system_default)
        self.create_subscription(Image, '/depth_image', self.depth_callback, qos_profile_system_default)
        self.pc_pub = self.create_publisher(PointCloud2, '/raw_point_cloud', qos_profile_system_default)

    def camera_info_callback(self, msg):
        self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_model.fromCameraInfo(msg)

    def depth_callback(self, msg):
        if self.camera_model is None:
            self.get_logger().warn("Waiting for camera info...")
            return
        
        # Convert depth image to numpy
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        height, width = depth_image.shape
        self.get_logger().info(f'depth shape: {height}x{width}')
        
        # Generate 3D points
        points = []
        for v in range(height):
            for u in range(width):
                z = depth_image[v, u] / 1000.0  # Convert mm to meters if needed
                if z == 0:
                    continue
                x, y, _ = self.camera_model.projectPixelTo3dRay((u, v))
                x, y = x * z, y * z
                points.append((x, y, z))
        
        if not points:
            self.get_logger().warn("No valid depth points")
            return
        
        # Convert to PointCloud2 and publish
        header = msg.header
        cloud_msg = point_cloud2.create_cloud_xyz32(header, points)
        self.pc_pub.publish(cloud_msg)
        self.get_logger().info("Published raw point cloud")


def main():
    rclpy.init()
    node = PillarDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
