import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
import cv2
import numpy as np
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_system_default
from message_filters import ApproximateTimeSynchronizer, Subscriber

class T265Tracker(Node):
    def __init__(self):
        super().__init__('realsense_tracking')
        self.bridge = CvBridge()
        
        # Camera intrinsics
        self.K_left = self.D_left = self.P_left = None
        self.K_right = self.D_right = self.P_right = None
        self.img_left = self.img_right = None
        self.pose = None
        
        # Create subscribers with message filters
        self.left_info_sub = Subscriber(self, CameraInfo, '/camera/fisheye1/camera_info')
        self.right_info_sub = Subscriber(self, CameraInfo, '/camera/fisheye2/camera_info')
        self.left_image_sub = Subscriber(self, Image, '/camera/fisheye1/image_raw')
        self.right_image_sub = Subscriber(self, Image, '/camera/fisheye2/image_raw')
        self.pose_sub = Subscriber(self, Odometry, '/camera/pose/sample')
        
        # Synchronizer
        self.sync = ApproximateTimeSynchronizer(
            [self.left_info_sub, self.right_info_sub, self.left_image_sub, self.right_image_sub], 
            queue_size=10, 
            slop=0.05
        )
        self.sync.registerCallback(self.sync_callback)
        
        # Publishers
        self.left_image_pub = self.create_publisher(Image, '/left/image_raw', qos_profile_system_default)
        self.left_info_pub = self.create_publisher(CameraInfo, '/left/camera_info', qos_profile_system_default)
        self.right_image_pub = self.create_publisher(Image, '/right/image_raw', qos_profile_system_default)
        self.right_info_pub = self.create_publisher(CameraInfo, '/right/camera_info', qos_profile_system_default)

    def sync_callback(self, left_info, right_info, left_img, right_img):
        """Processes synchronized image and camera info messages."""
        self.K_left = list(left_info.k) #np.array(left_info.k).reshape(3, 3)
        self.D_left = list(left_info.d) #np.array(left_info.d)
        self.P_left = list(left_info.p) #np.array(left_info.p).reshape(3, 4)
        
        self.K_right = list(right_info.k)  # np.array(right_info.k).reshape(3, 3)
        self.D_right = list(right_info.d) #np.array(right_info.d)
        self.P_right = list(right_info.p) #np.array(right_info.p).reshape(3, 4)
        
        self.img_left = self.bridge.imgmsg_to_cv2(left_img, desired_encoding='mono8')
        self.img_right = self.bridge.imgmsg_to_cv2(right_img, desired_encoding='mono8')
        
        self.get_logger().info(f"Synchronized images at {left_img.header.stamp.sec}.{right_img.header.stamp.sec}.{left_info.header.stamp.sec}.{right_info.header.stamp.sec}")
        left_info.distortion_model = 'plumb_bob'
        right_info.distortion_model = 'plumb_bob'
        self.left_image_pub.publish(left_img)
        self.right_image_pub.publish(right_img)
        self.left_info_pub.publish(left_info)
        self.right_info_pub.publish(right_info)

    def pose_callback(self, msg):
        """Extract camera position & orientation in world frame."""
        self.pose = {
            'position': (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z),
            'orientation': (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        }
        # self.get_logger().info(f"Updated pose: {self.pose}")


def main(args=None):
    rclpy.init(args=args)
    tracker = T265Tracker()
    rclpy.spin(tracker)
    tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
