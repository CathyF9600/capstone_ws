import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros
from rclpy.qos import qos_profile_system_default
import tf_transformations

def pixel_to_world(u, v, depth, K, R, t, pose):
    # Convert pixel to camera frame
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    X_c = (u - cx) * depth / fx
    Y_c = (v - cy) * depth / fy
    Z_c = depth

    # Camera frame coordinates
    p_c = np.array([X_c, Y_c, Z_c])

    # Get camera pose
    X_w, Y_w, Z_w = pose['position']
    qx, qy, qz, qw = pose['orientation']

    # Convert quaternion to rotation matrix
    R = tf_transformations.quaternion_matrix([qx, qy, qz, qw])[:3, :3]

    # Transform to world frame
    P_w = R @ p_c + np.array([X_w, Y_w, Z_w])
    return P_w


class DepthToPointCloud(Node):
    def __init__(self):
        super().__init__('realsense_tracking')
        self.bridge = CvBridge()
        self.pose = {'position':None, 'orientation':None}
        # Camera parameters (to be updated from CameraInfo)
        self.fx, self.fy, self.cx, self.cy, self.baseline = None, None, None, None, None
        self.K, self.P = None, None
        # Subscribers
        self.create_subscription(CameraInfo, '/camera/fisheye1/camera_info', self.camera_info_callback, qos_profile_system_default)
        self.create_subscription(Image, '/depth_image', self.depth_callback, qos_profile_system_default)
        # RealSense Subscriber
        self.realsense_sub = self.create_subscription(
            Odometry,
            "/camera/pose/sample",
            self.realsense_callback,
            qos_profile_system_default
        )
        # Publisher
        self.pc_pub = self.create_publisher(PointCloud2, '/point_cloud', qos_profile_system_default)

        # TF2 Buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def realsense_callback(self, msg):
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

    def camera_info_callback(self, msg):
        """Retrieve camera intrinsics from CameraInfo."""
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.baseline = 0.064  # 64mm stereo baseline (update as needed)
        self.K = np.array(msg.k).reshape(3,3)
        self.P = np.array(msg.p).reshape(3,4)

    def depth_callback(self, msg):
        """Convert depth image to point cloud."""
        if None in (self.fx, self.fy, self.cx, self.cy, self.baseline):
            return  # Wait for camera info

        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        height, width = depth_image.shape

        points = []
        for v in range(height):
            for u in range(width):
                disparity = depth_image[v, u]
                x,y,z = pixel_to_world(u, v, disparity, self.K, self.P[:, :3], self.P[:, 3], self.pose)
                # if disparity > 0:
                #     Z = (self.fx * self.baseline) / disparity
                #     X = (u - self.cx) * Z / self.fx
                #     Y = (v - self.cy) * Z / self.fy
                self.get_logger().info(f'world coord {type(x.item())} {y} {z}')
                points.append((x.item(), y.item(), z.item()))

        # Convert to PointCloud2
        header = msg.header
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        pc_msg = pc2.create_cloud(header, fields, points)

        # try:
        #     transform = self.tf_buffer.lookup_transform("map", msg.header.frame_id, rclpy.time.Time()) # to world frame
        #     pc_msg = self.transform_pointcloud(pc_msg, transform)
        # except Exception as e:
        #     self.get_logger().warn(f"Transform error: {e}")

        pc_msg.header.frame_id = 'odom_frame'
        # pc_msg.header.stamp = self.get_clock().now().to_msg()
        self.pc_pub.publish(pc_msg)


    def transform_pointcloud(self, cloud, transform):
        """Apply TF2 transform to PointCloud2."""
        points = list(pc2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True))
        transformed_points = []

        # Extract transform matrix
        translation = np.array([transform.transform.translation.x,
                                transform.transform.translation.y,
                                transform.transform.translation.z])
        rotation = np.array([transform.transform.rotation.x,
                             transform.transform.rotation.y,
                             transform.transform.rotation.z,
                             transform.transform.rotation.w])
        rotation_matrix = tf_transformations.quaternion_matrix(rotation)[:3, :3]

        for p in points:
            transformed_p = np.dot(rotation_matrix, np.array(p)) + translation
            transformed_points.append(tuple(transformed_p))

        return pc2.create_cloud(cloud.header, cloud.fields, transformed_points)

def main(args=None):
    rclpy.init(args=args)
    node = DepthToPointCloud()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()