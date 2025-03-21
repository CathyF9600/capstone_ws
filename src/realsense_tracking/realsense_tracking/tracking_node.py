import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
import cv2
import numpy as np
from cv_bridge import CvBridge

class T265Tracker(Node):
    def __init__(self):
        super().__init__('realsense_tracking')
        self.bridge = CvBridge()

        # Camera parameters (initialized later)
        self.K = None
        self.D = None
        self.new_K = None
        self.map1, self.map2 = None, None

        # Subscribe to camera info
        self.create_subscription(CameraInfo, '/camera/fisheye1/camera_info', self.camera_info_callback, 10)

        # Subscribe to fisheye images
        self.create_subscription(Image, '/camera/fisheye1/image_raw', self.left_callback, 10)
        self.create_subscription(Image, '/camera/fisheye2/image_raw', self.right_callback, 10)

        # Subscribe to odometry
        self.create_subscription(Odometry, '/camera/pose/sample', self.odom_callback, 10)

    def camera_info_callback(self, msg):
        """ Extract camera parameters when received. """
        self.K = np.array(msg.k).reshape(3, 3)
        self.D = np.array(msg.d)

        # Get image size
        w, h = msg.width, msg.height

        # Compute optimal new camera matrix
        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K, self.D, (w, h), np.eye(3), balance=0.0)

        # Precompute undistortion maps
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.new_K, (w, h), cv2.CV_16SC2)

        self.get_logger().info(f"Camera Intrinsics Loaded: fx={self.K[0,0]}, fy={self.K[1,1]}")

    def undistort_image(self, img):
        """ Applies fisheye undistortion. """
        if self.map1 is not None and self.map2 is not None:
            return cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
        return img  # If calibration data not received yet, return as-is

    def left_callback(self, msg):
        """ Process left fisheye camera. """
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        img = self.undistort_image(img)  # Apply undistortion
        cv2.imshow("Left Fisheye Undistorted", img)
        cv2.waitKey(1)

    def right_callback(self, msg):
        """ Process right fisheye camera. """
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        img = self.undistort_image(img)  # Apply undistortion
        cv2.imshow("Right Fisheye Undistorted", img)
        cv2.waitKey(1)

    def odom_callback(self, msg):
        """ Get global pose information. """
        pos = msg.pose.pose.position
        self.get_logger().info(f"Position: x={pos.x:.2f}, y={pos.y:.2f}, z={pos.z:.2f}")

def main():
    rclpy.init()
    node = T265Tracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
