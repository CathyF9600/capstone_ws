#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import os
from message_filters import ApproximateTimeSynchronizer, Subscriber
from datetime import datetime

class RGBDRecorder(Node):
    def __init__(self):
        super().__init__('rgbd_recorder')
        self.bridge = CvBridge()
        self.last_saved_time = self.get_clock().now()

        # Subscribers with time sync
        self.rgb_sub = Subscriber(self, Image, '/rgb_image')
        self.depth_sub = Subscriber(self, Image, '/depth_image')
        self.pose_sub = Subscriber(self, Odometry, '/camera/pose/sample')
        # self.vicon_sub = Subscriber(self, PoseStamped, "/vicon/ROB498_Drone/ROB498_Drone")

        self.sync = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub, self.pose_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.callback)

        self.output_dir = 'rgbd_npy_mock'
        os.makedirs(self.output_dir, exist_ok=True)

    def callback(self, rgb_msg, depth_msg, pose_msg):
        now = self.get_clock().now()
        time_since_last = (now - self.last_saved_time).nanoseconds / 1e9
        print(f'{rgb_msg.header.stamp.sec}, {depth_msg.header.stamp.sec}')
        if time_since_last < 0.1:
            return  # Only save every 1 second

        try:
            print('synched')
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')  # HxWx3
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')  # HxW
            pose = np.array([
                pose_msg.pose.pose.orientation.x,
                pose_msg.pose.pose.orientation.y,
                pose_msg.pose.pose.orientation.z,
                pose_msg.pose.pose.orientation.w,
                pose_msg.pose.pose.position.x,
                pose_msg.pose.pose.position.y,
                pose_msg.pose.pose.position.z
            ])
            # vicon_pose = np.array([vicon_msg.pose.position.x,
            #                        vicon_msg.pose.position.y,
            #                        vicon_msg.pose.position.z])
            if depth_image.ndim == 3:
                depth_image = depth_image[:, :, 0]  # If it's 3D, collapse to 2D

            h, w = depth_image.shape
            rgb_resized = cv2.resize(rgb_image, (w, h))

            rgbd = np.dstack((rgb_resized, depth_image))  # HxWx4
            DATA = {
                'rgbd': rgbd,
                'pose': pose,
                # 'vicon': vicon_pose
            }
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f'rgbd_{timestamp}.npy')
            np.save(filename, DATA)
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
