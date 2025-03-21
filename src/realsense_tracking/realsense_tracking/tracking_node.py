#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import cv2
import numpy as np
from cv_bridge import CvBridge
import tf_transformations
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy, qos_profile_system_default


class T265Tracker(Node):
    def __init__(self):
        super().__init__('realsense_tracking')
        self.bridge = CvBridge()

        # Subscribe to Fisheye Camera
        self.create_subscription(Image, '/camera/fisheye1/image_raw', self.image_callback, qos_profile_system_default)
        self.create_subscription(CameraInfo, '/camera/fisheye1/camera_info', self.camera_info_callback, qos_profile_system_default)

        # Subscribe to Camera Pose
        self.create_subscription(Odometry, '/camera/pose/sample', self.pose_callback, qos_profile_system_default)

        # Camera intrinsics
        self.fx = self.fy = self.cx = self.cy = None
        self.pose = None  # Store latest camera pose

    def camera_info_callback(self, msg):
        """Extract camera intrinsic parameters."""
        self.fx, self.fy = msg.k[0], msg.k[4]  # Focal lengths
        self.cx, self.cy = msg.k[2], msg.k[5]  # Principal point
        
        # self.get_logger().info(f"msg {self.fx}, {self.fy}, {self.cx}, {self.cy}")

    def pose_callback(self, msg):
        """Extract camera position & orientation in world frame."""
        self.pose = {
            'position': (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z),
            'orientation': (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        }
    def undistort_image(self, img):
        """ Applies fisheye undistortion. """
        if self.map1 is not None and self.map2 is not None:
            return cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
        return img  # If calibration data not received yet, return as-is

    def image_callback(self, msg):
        """Process incoming images, detect an object, and estimate its global position."""
        if self.fx is None or self.pose is None:
            self.get_logger().warn("Waiting for camera intrinsics and pose data...")
            return

        # Convert ROS image message to OpenCV format
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        img = self.undistort_image(img)
        # Dummy object detection (center of the image)
        u, v = int(img.shape[1] / 2), int(img.shape[0] / 2)

        # Convert pixel to world coordinates
        world_pos = self.pixel_to_world((u, v))

        # Display result
        text = f"X: {world_pos[0]:.2f}, Y: {world_pos[1]:.2f}, Z: {world_pos[2]:.2f}m"
        cv2.putText(img, text, (u - 50, v - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.circle(img, (u, v), 5, (255, 255, 255), -1)
        cv2.imshow("Tracked Image", img)
        cv2.waitKey(1)

    def pixel_to_world(self, pixel):
        """Convert image pixel to global world coordinates."""
        u, v = pixel
        X_c = (u - self.cx) / self.fx
        Y_c = (v - self.cy) / self.fy
        Z_c = 1  # Assume unit depth (scaling factor is unknown)

        # Camera frame coordinates
        P_c = np.array([X_c, Y_c, Z_c])

        # Get camera pose
        X_w, Y_w, Z_w = self.pose['position']
        qx, qy, qz, qw = self.pose['orientation']

        # Convert quaternion to rotation matrix
        R = tf_transformations.quaternion_matrix([qx, qy, qz, qw])[:3, :3]

        # Transform to world frame
        P_w = R @ P_c + np.array([X_w, Y_w, Z_w])
        return P_w

def main(args=None):
    rclpy.init(args=args)
    node = T265Tracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
