#!/usr/bin/env python3
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import cv2
import numpy as np
from cv_bridge import CvBridge
import tf_transformations
from rclpy.qos import QoSProfile, qos_profile_system_default
from std_msgs.msg import Float32MultiArray
import time
from message_filters import ApproximateTimeSynchronizer, Subscriber

WINDOW_TITLE = 'Realsense'
cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
window_size = 5
min_disp = 0
num_disp = 96 - min_disp
max_disp = min_disp + num_disp

# stereo camera constants
H, W = 800, 848
IMG_SIZE_WH = (W, H)
DOWNSCALE_H = 4
STEREO_SIZE_WH = (W, H//DOWNSCALE_H)
BASELINE = -18.2928466796875/286.1825866699219 # 64 mm baseline

BRIDGE = CvBridge()

# Add the flag for visualization choice
visualize_output = True

def get_user_input():
    global visualize_output
    print("Would you like to visualize the output? (y/n)")
    user_input = input().strip().lower()
    if user_input == 'n':
        visualize_output = False

# Ask for visualization preference at the beginning
get_user_input()

class T265Tracker(Node):
    def __init__(self):
        super().__init__('realsense_tracking')
        self.get_logger().info(f"T265 Node started!" )
        self.bridge = CvBridge()
        self.pose = {'position':None, 'orientation':None}

        # Create Disparity Publisher
        self.depth_pub = self.create_publisher(Image, '/depth_image', qos_profile_system_default)
        self.color_pub = self.create_publisher(Image, '/rgb_image', qos_profile_system_default)
        self.rgbd_pub = self.create_publisher(Float32MultiArray, '/rgbd_data', qos_profile_system_default)

        self.left_image_sub = Subscriber(self, Image, '/camera/fisheye1/image_raw')
        self.right_image_sub = Subscriber(self, Image, '/camera/fisheye2/image_raw')
        self.left_info_sub = self.create_subscription(CameraInfo, '/camera/fisheye1/camera_info', self.camera_info_callback_l, qos_profile_system_default)
        self.right_info_sub = self.create_subscription(CameraInfo, '/camera/fisheye2/camera_info', self.camera_info_callback_r, qos_profile_system_default)
        self.pose_sub = self.create_subscription(Odometry, '/camera/pose/sample', self.pose_callback, qos_profile_system_default)

        # Synchronizer
        self.sync = ApproximateTimeSynchronizer(
            [self.left_image_sub, self.right_image_sub], 
            queue_size=10,
            slop=0.001
        )
        self.sync.registerCallback(self.sync_callback)
        self.timer = self.create_timer(0.02, self.bm)

    def sync_callback(self, img_msg1, img_msg2):
        self.stack.append((img_msg1, img_msg2))

    def bm(self):
        if not self.stack:
            return
        img_msg1, img_msg2 = self.stack.pop()

        if not self.camera_info_msg1 or not self.camera_info_msg2:
            return

        # Calculate disparity
        disparity = self.compute_disparity(img_msg1, img_msg2)

        # Generate RGBD image
        depth = (self.fx * -BASELINE) / (disparity + 1e-6)
        rgbd = self.create_rgbd_image(depth, img_msg1)

        msg = Float32MultiArray()
        msg.data = rgbd.flatten().tolist()
        self.rgbd_pub.publish(msg)

        # Publish depth and color images
        self.publish_depth_and_color_images(depth, rgbd)

        # Visualize only if the flag is set
        if visualize_output:
            self.visualize_output(depth, disparity, rgbd)

    def compute_disparity(self, img_msg1, img_msg2):
        img_distorted1 = BRIDGE.imgmsg_to_cv2(img_msg1, desired_encoding="mono8")
        img_distorted2 = BRIDGE.imgmsg_to_cv2(img_msg2, desired_encoding="mono8")

        # Rectify images
        img_undistorted1 = cv2.remap(
            img_distorted1, MAPX1, MAPY1, interpolation=cv2.INTER_LINEAR
        )
        img_undistorted2 = cv2.remap(
            img_distorted2, MAPX2, MAPY2, interpolation=cv2.INTER_LINEAR
        )

        # Compute disparity
        disparity = stereo.compute(img_undistorted1, img_undistorted2).astype(np.float32) / 16.0
        return disparity

    def create_rgbd_image(self, depth, img_msg1):
        color_image = cv2.cvtColor(img_msg1, cv2.COLOR_GRAY2RGB)
        depth_image = np.expand_dims(depth, axis=-1)
        rgbd = np.concatenate([color_image, depth_image], axis=-1)
        return rgbd

    def publish_depth_and_color_images(self, depth, rgbd):
        depth_msg = BRIDGE.cv2_to_imgmsg(depth, encoding="32FC1")
        depth_msg.header.stamp = self.get_clock().now().to_msg()
        self.depth_pub.publish(depth_msg)

        color_msg = BRIDGE.cv2_to_imgmsg(rgbd[:, :, :3], encoding="rgb8")
        color_msg.header.stamp = self.get_clock().now().to_msg()
        self.color_pub.publish(color_msg)

    def visualize_output(self, depth, disparity, rgbd):
        disp_vis = 255*(disparity - min_disp) / num_disp
        disp_color = cv2.applyColorMap(cv2.convertScaleAbs(disp_vis,1), cv2.COLORMAP_JET)

        # Display result
        cv2.imshow(WINDOW_TITLE, np.hstack((rgbd[:, :, :3], disp_color)))
        cv2.waitKey(1)

    def camera_info_callback_l(self, msg):
        self.camera_info_msg1 = msg

    def camera_info_callback_r(self, msg):
        self.camera_info_msg2 = msg

# Initialize the node and spin
def main(args=None):
    rclpy.init(args=args)
    t265_tracker = T265Tracker()
    rclpy.spin(t265_tracker)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
