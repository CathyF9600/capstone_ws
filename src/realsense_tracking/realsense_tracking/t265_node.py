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
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy, qos_profile_system_default
from math import tan, pi
import time
from message_filters import ApproximateTimeSynchronizer, Subscriber


WINDOW_TITLE = 'Realsense'
cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
window_size = 5
min_disp = 0
# must be divisible by 16
num_disp = 112 - min_disp
max_disp = min_disp + num_disp


# stereo camera constants
H, W = 800, 848
IMG_SIZE_WH = (W, H)
DOWNSCALE_H = 4
STEREO_SIZE_WH = (W, H//DOWNSCALE_H)
BASELINE = -18.2928466796875/286.1825866699219 # 64 mm baseline
DROP_FRAMES = 3

MAPX1 = None
MAPY1 = None
MAPX2 = None
MAPY2 = None

CAM_INFO1_ORIGINAL = None
CAM_INFO2_ORIGINAL = None
CAM_INFO1_MODIFIED = None
CAM_INFO2_MODIFIED = None
BRIDGE = CvBridge()

DROP_IND = 0


stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                        numDisparities = num_disp,
                        blockSize = 21,  # Increased block size # 16
                        P1 = 8*3*window_size**2,
                        P2 = 32*3*window_size**2,
                        disp12MaxDiff = 1,
                        uniquenessRatio = 10,
                        speckleWindowSize = 100,
                        speckleRange = 32)

class T265Tracker(Node):
    def __init__(self):
        super().__init__('realsense_tracking')
        self.bridge = CvBridge()
        self.pose = {'position':None, 'orientation':None}

        # Create Disparity Publisher
        self.disparity_pub = self.create_publisher(Image, '/depth_image', qos_profile_system_default)

        # Camera intrinsics
        self.fx = self.fy = self.cx = self.cy = None

        self.K_left = None
        self.D_left = None
        self.P_left = None
        self.img_left = None

        self.K_right = None
        self.D_right = None
        self.P_right = None
        self.img_right = None

        self.R_rel = None
        self.T_rel = None

        self.lm1, self.lm2 = None, None
        self.rm1, self.rm2 = None, None

        self.prevT = 0

        self.undistort_rectify = None

        self.lock = threading.Lock()

        # Create subscribers with message filters
        self.left_info_sub = Subscriber(self, CameraInfo, '/camera/fisheye1/camera_info')
        self.right_info_sub = Subscriber(self, CameraInfo, '/camera/fisheye2/camera_info')
        self.left_image_sub = Subscriber(self, Image, '/camera/fisheye1/image_raw')
        self.right_image_sub = Subscriber(self, Image, '/camera/fisheye2/image_raw')
        self.pose_sub = self.create_subscription(Odometry, '/camera/pose/sample', self.pose_callback,
            qos_profile_system_default)
        # Synchronizer
        self.sync = ApproximateTimeSynchronizer(
            [self.left_image_sub, self.right_image_sub, self.left_info_sub, self.right_info_sub], 
            queue_size=10,
            slop=0.001
        )
        self.sync.registerCallback(self.sync_callback)
    
    def sync_callback(self, img_msg1, img_msg2, camera_info_msg1, camera_info_msg2):
        global DROP_IND

        if any([x is None for x in (MAPX1, MAPY1, MAPX2, MAPY2)]):
            self.init_maps(camera_info_msg1, camera_info_msg2)

        img_distorted1 = BRIDGE.imgmsg_to_cv2(img_msg1, desired_encoding="mono8")
        img_distorted2 = BRIDGE.imgmsg_to_cv2(img_msg2, desired_encoding="mono8")

        img_undistorted1 = cv2.remap(
            img_distorted1,
            MAPX1,
            MAPY1,
            interpolation=cv2.INTER_LINEAR,
        )

        img_undistorted2 = cv2.remap(
            img_distorted2,
            MAPX2,
            MAPY2,
            interpolation=cv2.INTER_LINEAR,
        )

        # disparity_full = stereo.compute(img_undistorted1, img_undistorted2).astype(np.float32) / 16.0
        # disparity_blur = cv2.medianBlur(disparity_full, 5)  # Added median filter
        # # Publish disparity msgs
        # disp_msg = BRIDGE.cv2_to_imgmsg(disparity_blur, encoding="32FC1")
        # disp_msg.header.stamp = self.get_clock().now().to_msg()
        # self.disparity_pub.publish(disp_msg)
        # self.get_logger().info(f'R: {camera_info_msg1.p}')

        # crop top and bottom based on DOWNSCALE_H
        orig_height = img_undistorted1.shape[0]
        new_height = orig_height//DOWNSCALE_H

        # take center of image of new height
        img_undistorted1 = img_undistorted1[
            (orig_height - new_height)//2 : (orig_height + new_height)//2, :
        ]
        img_undistorted2 = img_undistorted2[
            (orig_height - new_height)//2 : (orig_height + new_height)//2, :
        ]
        
        # # convert from mono8 to bgr8
        # img_undistorted1 = cv2.cvtColor(img_undistorted1, cv2.COLOR_GRAY2BGR)
        # output_msg1 = BRIDGE.cv2_to_imgmsg(img_undistorted1, encoding="bgr8")
        # output_msg1.header = img_msg1.header
        # img_undistorted2 = cv2.cvtColor(img_undistorted2, cv2.COLOR_GRAY2BGR)
        # output_msg2 = BRIDGE.cv2_to_imgmsg(img_undistorted2, encoding="bgr8")
        # output_msg2.header = img_msg2.header

        # # publish
        # self.left_image_pub.publish(output_msg1)
        # self.right_image_pub.publish(output_msg2)
        # self.left_info_pub.publish(camera_info_msg1)
        # self.right_info_pub.publish(camera_info_msg2)
        mode = "stack"
        # compute the disparity on the center of the frames and convert it to a pixel disparity (divide by DISP_SCALE=16)
        disparity = stereo.compute(img_undistorted1, img_undistorted2).astype(np.float32) / 16.0
        disparity_blur = cv2.medianBlur(disparity, 5)  # Added median filter

        # re-crop just the valid part of the disparity
        disparity = disparity_blur[:,max_disp:]

        # Publish disparity msgs
        disp_msg = BRIDGE.cv2_to_imgmsg(disparity, encoding="32FC1")
        disp_msg.header.stamp = self.get_clock().now().to_msg()
        self.disparity_pub.publish(disp_msg)

        # convert disparity to 0-255 and color it
        disp_vis = 255*(disparity - min_disp)/ num_disp
        disp_color = cv2.applyColorMap(cv2.convertScaleAbs(disp_vis,1), cv2.COLORMAP_JET)
        color_image = cv2.cvtColor(img_undistorted1[:,max_disp:], cv2.COLOR_GRAY2RGB)
        # self.get_logger().info(f"d {disp_color}")
        u, v = int(img_undistorted2.shape[1] / 2), int(img_undistorted2.shape[0] / 2)
        
        self.get_logger().info(f"Synchronized images at {img_msg1.header.stamp.sec}.{img_msg2.header.stamp.sec}.{camera_info_msg1.header.stamp.sec}.{camera_info_msg2.header.stamp.sec}")
        now = self.get_clock().now().to_msg().sec / 1e3
        bm_latency = now - img_msg1.header.stamp.sec / 1e3
        # self.prevT = now
        self.get_logger().info(f"BM latency: {bm_latency:.4f} ms" )
        fx_l = camera_info_msg1.k[0]
        depth = (fx_l * -BASELINE) / (disparity + 1e-6)

        if mode == "stack":
            # cv2.circle(img_undistorted2, (u, v), 5, (255, 255, 255), -1)
            # # cv2.imshow("Tracked Image", color_image)
            # cv2.waitKey(1)
            # # Display result
            text = f"depth: {depth.shape}, {depth[v][u]}, {u}, {v}"
            # text = ''
            cv2.putText(color_image, text, (u - 50, v - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(color_image, (u, v), 10, (255, 255, 255), -1)
            cv2.imshow(WINDOW_TITLE, np.hstack((color_image, disp_color)))
            # cv2.imshow("Tracked Image", img_undistorted2)
            cv2.waitKey(1)

    def init_maps(self, cam_info1, cam_info2):
        global MAPX1, MAPY1, MAPX2, MAPY2
        K1 = np.array(cam_info1.k).reshape(3,3)
        D1 = np.array(cam_info1.d)
        K2 = np.array(cam_info2.k).reshape(3,3)
        D2 = np.array(cam_info2.d)
        T = np.array([BASELINE, 0, 0]) # 64 mm baseline

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, IMG_SIZE_WH, R=np.eye(3), T=T
        )
        MAPX1, MAPY1 = cv2.fisheye.initUndistortRectifyMap(
            K1, D1, R1, P1, size=IMG_SIZE_WH, m1type=cv2.CV_32FC1
        )
        MAPX2, MAPY2 = cv2.fisheye.initUndistortRectifyMap(
            K2, D2, R2, P2, size=IMG_SIZE_WH, m1type=cv2.CV_32FC1
        )

    def get_extrinsics(self, P_left, P_right):
        # Returns R, T transform from src (left) to dst
        # Extract R and T from 3x4 projection matrix (P = K [R | T])
        R_left, T_left = P_left[:, :3], P_left[:, 3]
        R_right, T_right = P_right[:, :3], P_right[:, 3]
        
        # Compute relative transformation
        self.R_rel = R_right @ np.linalg.inv(R_left)
        self.T_rel = T_right - self.R_rel @ T_left
        self.get_logger().info(f"R, T {self.R_rel} {self.T_rel}")

    def camera_info_callback_l(self, msg):
        """Extract camera intrinsic parameters."""
        self.fx, self.fy = msg.k[0], msg.k[4]  # Focal lengths
        self.cx, self.cy = msg.k[2], msg.k[5]  # Principal point
        
        self.K_left = np.array(msg.k).reshape(3, 3)
        self.D_left = np.array(msg.d)
        self.P_left = np.array(msg.p).reshape(3, 4)

        # Get image size
        w, h = msg.width, msg.height
        # Precompute undistortion maps L 
        # self.lm1, self.lm2 = cv2.fisheye.initUndistortRectifyMap(self.K_left, self.D_left, np.eye(3), self.P_left, (w, h), cv2.CV_32FC1)
        self.get_logger().info(f"msg L {self.fx}, {self.fy}, {self.cx}, {self.cy}")

    def camera_info_callback_r(self, msg):
        """Extract camera intrinsic parameters."""
        self.fx, self.fy = msg.k[0], msg.k[4]  # Focal lengths
        self.cx, self.cy = msg.k[2], msg.k[5]  # Principal point
        
        self.K_right = np.array(msg.k).reshape(3, 3)
        self.D_right = np.array(msg.d)
        self.P_right = np.array(msg.p).reshape(3, 4)
        if (isinstance(self.P_left, np.ndarray) and isinstance(self.P_right, np.ndarray)):
            # self.get_logger().info(f"P_left {self.P_left}, {self.P_right}")
            if not (isinstance(self.R_rel, np.ndarray) and isinstance(self.T_rel, np.ndarray)):
                self.get_extrinsics(self.P_left, self.P_right) # get relative R, T
            # Get image size
            w, h = msg.width, msg.height

            # self.get_logger().info(f"msg R {self.fx}, {self.fy}, {self.cx}, {self.cy}")
        else:
            self.get_logger().warn(f"Waiting for P matrices")


    def pose_callback(self, msg):
        """Extract camera position & orientation in world frame."""
        self.pose['position'] = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        # Original orientation
        self.pose['orientation'] = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])

        # self.get_logger().info(f"Cam ori: x={msg.pose.pose.orientation.x:.3f}, y={msg.pose.pose.orientation.y:.3f}, z={msg.pose.pose.orientation.z:.3f}, \
        #                         w={msg.pose.pose.orientation.w:.3f}")
        # self.get_logger().info(f"Cam pos: x={msg.pose.pose.position.x:.3f}, y={msg.pose.pose.position.y:.3f}, \
        #                         z={msg.pose.pose.position.z:.3f}")

    def image_callback_l(self, msg):
        """Process incoming images, detect an object, and estimate its global position."""
        if self.fx is None or self.pose is None:
            self.get_logger().warn("Waiting for camera intrinsics and pose data...")
            return
        
        # Acquire lock to access shared resources safely
        with self.lock:
            # Convert ROS image message to OpenCV format
            self.img_left = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        # img = self.undistort_image(img)
        # # Dummy object detection (center of the image)
        # u, v = int(self.img_left.shape[1] / 2), int(self.img_left.shape[0] / 2)

        # # Convert pixel to world coordinates
        # world_pos = self.pixel_to_world((u, v))

        # # Display result
        # text = f"X: {world_pos[0]:.2f}, Y: {world_pos[1]:.2f}, Z: {world_pos[2]:.2f}m"
        # cv2.putText(self.img_left, text, (u - 50, v - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # cv2.circle(self.img_left, (u, v), 5, (255, 255, 255), -1)
        # cv2.imshow("Tracked Image", self.img_left)
        # cv2.waitKey(1)

    def image_callback_r(self, msg):
        """Process incoming images, detect an object, and estimate its global position."""
        if self.fx is None or self.pose is None:
            self.get_logger().warn("Waiting for camera intrinsics and pose data...")
            return
        
        with self.lock:
            # Convert ROS image message to OpenCV format
            self.img_right = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            # img = self.undistort_image(img)
            # Dummy object detection (center of the image)
            # u, v = int(self.img_right.shape[1] / 2), int(self.img_right.shape[0] / 2)

            # # Convert pixel to world coordinates
            # world_pos = self.pixel_to_world((u, v))

            # # Display result
            # text = f"X: {world_pos[0]:.2f}, Y: {world_pos[1]:.2f}, Z: {world_pos[2]:.2f}m"
            # cv2.putText(self.img_right, text, (u - 50, v - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # cv2.circle(self.img_right, (u, v), 5, (255, 255, 255), -1)
            # cv2.imshow("Tracked Image", self.img_right)
            # cv2.waitKey(1)

            # We need to determine what focal length our undistorted images should have
            # in order to set up the camera matrices for initUndistortRectifyMap.  We
            # could use stereoRectify, but here we show how to derive these projection
            # matrices from the calibration and a desired height and field of view

            # We calculate the undistorted focal length:
            #
            #         h
            # -----------------
            #  \      |      /
            #    \    | f  /
            #     \   |   /
            #      \ fov /
            #        \|/
            stereo_fov_rad = 90 * (pi/180)  # 90 degree desired fov
            stereo_height_px = 300          # 300x300 pixel stereo output
            stereo_focal_px = stereo_height_px/2 / tan(stereo_fov_rad/2)

            # We set the left rotation to identity and the right rotation
            # the rotation between the cameras
            if (isinstance(self.R_rel, np.ndarray) and isinstance(self.T_rel, np.ndarray) and isinstance(self.img_left, np.ndarray) and isinstance(self.img_right, np.ndarray)):
                R_left = np.eye(3)
                R_right = self.R_rel

                # The stereo algorithm needs max_disp extra pixels in order to produce valid
                # disparity on the desired output region. This changes the width, but the
                # center of projection should be on the center of the cropped image
                stereo_width_px = stereo_height_px + max_disp
                stereo_size = (stereo_width_px, stereo_height_px)
                stereo_cx = (stereo_height_px - 1)/2 + max_disp
                stereo_cy = (stereo_height_px - 1)/2

                # Construct the left and right projection matrices, the only difference is
                # that the right projection matrix should have a shift along the x axis of
                # baseline*focal_length
                P_left = np.array([[stereo_focal_px, 0, stereo_cx, 0],
                                [0, stereo_focal_px, stereo_cy, 0],
                                [0,               0,         1, 0]])
                P_right = P_left.copy()
                P_right[0][3] = self.T_rel[0]*stereo_focal_px

                # P_left = self.P_left
                # P_right = P_left.copy()
                # P_right[0][3] = self.T_rel[0]*stereo_focal_px

                # Create an undistortion map for the left and right camera which applies the
                # rectification and undoes the camera distortion. This only has to be done
                # once
                m1type = cv2.CV_32FC1
                if self.K_left is None:
                    self.get_logger().warn("K_left can't be None.")
                    return
                (lm1, lm2) = cv2.fisheye.initUndistortRectifyMap(self.K_left, self.D_left, R_left, P_left, stereo_size, m1type)
                (rm1, rm2) = cv2.fisheye.initUndistortRectifyMap(self.K_right, self.D_right, R_right, P_right, stereo_size, m1type)
                undistort_rectify = {"left"  : (lm1, lm2),
                                    "right" : (rm1, rm2)}
                # the rotation between the cameras
                mode = "stack"
                frame_copy = {"left"  : self.img_left.copy(),
                            "right" : self.img_right.copy()}
                # Undistort and crop the center of the frames
                center_undistorted = {"left" : cv2.remap(src = frame_copy["left"],
                                                map1 = undistort_rectify["left"][0],
                                                map2 = undistort_rectify["left"][1],
                                                interpolation = cv2.INTER_LINEAR),
                                        "right" : cv2.remap(src = frame_copy["right"],
                                                map1 = undistort_rectify["right"][0],
                                                map2 = undistort_rectify["right"][1],
                                                interpolation = cv2.INTER_LINEAR)}

                # compute the disparity on the center of the frames and convert it to a pixel disparity (divide by DISP_SCALE=16)
                disparity = stereo.compute(center_undistorted["left"], center_undistorted["right"]).astype(np.float32) / 16.0

                # re-crop just the valid part of the disparity
                disparity = disparity[:,max_disp:]
                
                # convert disparity to 0-255 and color it
                disp_vis = 255*(disparity - min_disp)/ num_disp
                disp_color = cv2.applyColorMap(cv2.convertScaleAbs(disp_vis,1), cv2.COLORMAP_JET)
                color_image = cv2.cvtColor(center_undistorted["left"][:,max_disp:], cv2.COLOR_GRAY2RGB)
                # self.get_logger().info(f"d {disp_color}")
                if mode == "stack":
                    cv2.imshow(WINDOW_TITLE, np.hstack((color_image, disp_color)))
                    # cv2.circle(self.img_right, (u, v), 5, (255, 255, 255), -1)
                    # cv2.imshow("Tracked Image", color_image)
                    cv2.waitKey(1)
            
            else:
                self.get_logger().warn(f"Waiting for R_rel, T_rel matrices")

        
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