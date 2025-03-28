import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from collections import deque
import cv2
import numpy as np
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_system_default

class T265Tracker(Node):
    def __init__(self):
        super().__init__('realsense_tracking')
        self.bridge = CvBridge()
        
        # Queues for image synchronization
        self.left_queue = deque()
        self.right_queue = deque()
        self.left_info_queue = deque()
        self.right_info_queue = deque()
        
        # Camera intrinsics
        self.K_left = self.D_left = self.P_left = None
        self.K_right = self.D_right = self.P_right = None
        self.img_left = self.img_right = None
        self.pose = None
        
        # Subscribe to Camera Info
        self.create_subscription(CameraInfo, '/camera/fisheye1/camera_info', self.camera_info_callback_l, qos_profile_system_default)
        self.create_subscription(CameraInfo, '/camera/fisheye2/camera_info', self.camera_info_callback_r, qos_profile_system_default)
        
        # Subscribe to Image Streams
        self.create_subscription(Image, '/camera/fisheye1/image_raw', self.image_callback_l, qos_profile_system_default)
        self.create_subscription(Image, '/camera/fisheye2/image_raw', self.image_callback_r, qos_profile_system_default)
        
        # Subscribe to Camera Pose
        self.create_subscription(Odometry, '/camera/pose/sample', self.pose_callback, qos_profile_system_default)
        
        # Publishers for rectified images and camera info
        self.left_image_pub = self.create_publisher(Image, '/left/image_raw', qos_profile_system_default)
        self.left_info_pub = self.create_publisher(CameraInfo, '/left/camera_info', qos_profile_system_default)
        self.right_image_pub = self.create_publisher(Image, '/right/image_raw', qos_profile_system_default)
        self.right_info_pub = self.create_publisher(CameraInfo, '/right/camera_info', qos_profile_system_default)

    def camera_info_callback_l(self, msg):
        """Store left camera info and check for synchronization."""
        self.K_left = np.array(msg.k).reshape(3, 3)
        self.D_left = np.array(msg.d)
        self.P_left = np.array(msg.p).reshape(3, 4)
        self.left_info_queue.append(msg)
        self.process_synchronized_images()

    def camera_info_callback_r(self, msg):
        """Store right camera info and check for synchronization."""
        self.K_right = np.array(msg.k).reshape(3, 3)
        self.D_right = np.array(msg.d)
        self.P_right = np.array(msg.p).reshape(3, 4)
        self.right_info_queue.append(msg)
        self.process_synchronized_images()

    def image_callback_l(self, msg):
        """Store left image and check for synchronization."""
        self.left_queue.append(msg)
        self.process_synchronized_images()

    def image_callback_r(self, msg):
        """Store right image and check for synchronization."""
        self.right_queue.append(msg)
        self.process_synchronized_images()

    def process_synchronized_images(self):
        """Match left and right images along with their camera info based on timestamp."""
        if not self.left_queue or not self.right_queue or not self.left_info_queue or not self.right_info_queue:
            return  # Wait for all queues to have messages

        left_msg = self.left_queue[0]
        right_msg = self.right_queue[0]
        left_info_msg = self.left_info_queue[0]
        right_info_msg = self.right_info_queue[0]

        time_diff_img = abs(left_msg.header.stamp.sec + left_msg.header.stamp.nanosec * 1e-9 -
                            right_msg.header.stamp.sec - right_msg.header.stamp.nanosec * 1e-9)

        time_diff_info = abs(left_info_msg.header.stamp.sec + left_info_msg.header.stamp.nanosec * 1e-9 -
                              right_info_msg.header.stamp.sec - right_info_msg.header.stamp.nanosec * 1e-9)
        time_diff_ci = abs(left_info_msg.header.stamp.sec + left_info_msg.header.stamp.nanosec * 1e-9 -
                              left_msg.header.stamp.sec - left_msg.header.stamp.nanosec * 1e-9)
        if time_diff_img < 0.01 and time_diff_info < 0.01 and time_diff_ci < 0.01 : 
            self.left_info_queue.popleft()
            self.right_info_queue.popleft()
            self.left_queue.popleft()
            self.right_queue.popleft()

            self.img_left = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='mono8')
            self.img_right = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='mono8')
            
            self.get_logger().info(f"Synchronized images and info at {left_msg.header.stamp.sec}.{right_msg.header.stamp.sec}")
            
            # Publish synchronized images and camera info
            self.left_image_pub.publish(left_msg)
            self.right_image_pub.publish(right_msg)
            self.left_info_pub.publish(left_info_msg)
            self.right_info_pub.publish(right_info_msg)

        elif left_info_msg.header.stamp.sec < right_info_msg.header.stamp.sec:
            self.left_info_queue.popleft()
        elif left_info_msg.header.stamp.sec > right_info_msg.header.stamp.sec:
            self.right_info_queue.popleft()

        elif left_msg.header.stamp.sec < right_msg.header.stamp.sec:
            self.left_queue.popleft()  # Drop old left image
        elif left_msg.header.stamp.sec > right_msg.header.stamp.sec:
            self.right_queue.popleft()  # Drop old right image
        
        # now lc-rc and li-ri are synchd
        #next we synch rc-ri
        
        elif left_info_msg.header.stamp.sec < left_msg.header.stamp.sec:
            self.left_info_queue.popleft()
            self.right_info_queue.popleft()

        elif left_info_msg.header.stamp.sec > left_msg.header.stamp.sec:
            self.left_queue.popleft() 
            self.right_queue.popleft() 

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
