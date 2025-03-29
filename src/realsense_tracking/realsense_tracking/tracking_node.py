import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
import cv2
import numpy as np
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_system_default
from message_filters import ApproximateTimeSynchronizer, Subscriber


# stereo camera constants
H, W = 800, 848
IMG_SIZE_WH = (W, H)
DOWNSCALE_H = 8
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

class T265Tracker(Node):
    def __init__(self):
        super().__init__('realsense_tracking')
        
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
            [self.left_image_sub, self.right_image_sub, self.left_info_sub, self.right_info_sub], 
            queue_size=10,
            slop=0.05
        )

        self.sync.registerCallback(self.sync_callback)
        
        # Publishers
        self.left_image_pub = self.create_publisher(Image, '/left/image_raw', qos_profile_system_default)
        self.left_info_pub = self.create_publisher(CameraInfo, '/left/camera_info', qos_profile_system_default)
        self.right_image_pub = self.create_publisher(Image, '/right/image_raw', qos_profile_system_default)
        self.right_info_pub = self.create_publisher(CameraInfo, '/right/camera_info', qos_profile_system_default)

    def sync_callback0(self, left_info, right_info, left_img, right_img):
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

    def sync_callback(self, img_msg1, img_msg2, camera_info_msg1, camera_info_msg2):
        global DROP_IND
        check_none = (
            self.left_image_pub,
            self.right_image_pub,
            self.left_info_pub,
            self.right_info_pub,
        )
        if any([x is None for x in check_none]):
            self.get_logger().warn("Waiting for camera info")
            return
        if any([x is None for x in (MAPX1, MAPY1, MAPX2, MAPY2)]):
            self.init_maps(camera_info_msg1, camera_info_msg2)
        # drop frames to reduce load
        DROP_IND += 1
        if DROP_IND < DROP_FRAMES:
            return
        DROP_IND = 0

        img_distorted1 = BRIDGE.imgmsg_to_cv2(img_msg1, desired_encoding="passthrough")
        img_undistorted1 = cv2.remap(
            img_distorted1,
            MAPX1,
            MAPY1,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        img_distorted2 = BRIDGE.imgmsg_to_cv2(img_msg2, desired_encoding="passthrough")
        img_undistorted2 = cv2.remap(
            img_distorted2,
            MAPX2,
            MAPY2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
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
        
        # convert from mono8 to bgr8
        img_undistorted1 = cv2.cvtColor(img_undistorted1, cv2.COLOR_GRAY2BGR)
        output_msg1 = BRIDGE.cv2_to_imgmsg(img_undistorted1, encoding="bgr8")
        output_msg1.header = img_msg1.header
        img_undistorted2 = cv2.cvtColor(img_undistorted2, cv2.COLOR_GRAY2BGR)
        output_msg2 = BRIDGE.cv2_to_imgmsg(img_undistorted2, encoding="bgr8")
        output_msg2.header = img_msg2.header

        # update camera info
        camera_info_msg1 = self.modify_camera_info(camera_info_msg1)
        camera_info_msg2 = self.modify_camera_info(camera_info_msg2)

        # publish
        self.left_image_pub.publish(output_msg1)
        # self.right_image_pub.publish(output_msg2)
        # self.left_info_pub.publish(camera_info_msg1)
        # self.right_info_pub.publish(camera_info_msg2)

        self.get_logger().info(f"Synchronized images at {output_msg1.header.stamp.sec}.{output_msg2.header.stamp.sec}.{camera_info_msg1.header.stamp.sec}.{camera_info_msg2.header.stamp.sec}")


    def modify_camera_info(self, msg):
        # print('msg', msg)
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        # downscale K and P
        msg.k = list(msg.k)
        msg.k[5] = msg.k[5]/DOWNSCALE_H # optical center
        msg.p = list(msg.p)
        msg.p[6] = msg.p[6]/DOWNSCALE_H # optical center
        msg.height = msg.height//DOWNSCALE_H
        return msg


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
