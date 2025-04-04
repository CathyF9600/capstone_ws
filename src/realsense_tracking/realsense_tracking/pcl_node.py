from sklearn.cluster import MiniBatchKMeans
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

def pixel_to_world(M, K, pose):
    # K: 4x4
    # M: Nx4 [u v d|1]
    Z_c = np.linalg.inv(K) @ M.T  # Preferred method

    # Get camera pose
    X_w, Y_w, Z_w = pose['position']
    qx, qy, qz, qw = pose['orientation']

    # Convert quaternion to rotation matrix
    R = tf_transformations.quaternion_matrix([qx, qy, qz, qw])[:3, :3]
    t = np.array([X_w, Y_w, Z_w])
    T = np.block([
        [R, t.reshape(3, 1)], 
        [np.zeros((1, 3)), 1]
    ])    # Transform to world frame
    P_w = T @ Z_c
    
    return P_w.T # (N, 3)



def pillarize_points_kmeans(points, n_clusters=200, bin_size=0.25, pillar_height=2.0):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=42)
    cluster_centers = kmeans.fit(points[:, :2]).cluster_centers_

    # Generate pillar points
    pillar_points = []
    for x, y in cluster_centers:
        for z in np.linspace(0, pillar_height, num=6):  # Vertical pillars
            pillar_points.append((x, y, z))

    return np.array(pillar_points)


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
        self.denoised_pub = self.create_publisher(PointCloud2, '/denoised_cloud', qos_profile_system_default)

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
        # self.K = np.array(msg.k).reshape(3,3)
        self.baseline = 0.064  # 64mm stereo baseline (update as needed)
        K = np.array(msg.k).reshape((3,3))
        self.K = np.block([
            [K, np.zeros((3, 1))], 
            [np.zeros((1, 3)), 1]
        ])
        self.P = np.array(msg.p).reshape(3,4)

    def depth_callback(self, msg):
        start = self.get_clock().now().to_msg().sec +  self.get_clock().now().to_msg().nanosec * 1e-9
        """Convert depth image to point cloud."""
        if None in (self.fx, self.fy, self.cx, self.cy, self.baseline):
            return  # Wait for camera info
        # self.get_logger().info(f'K shape: {self.K.shape}')

        # Depth Image Processing
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        height, width = depth_image.shape
        self.K[0][2] = width / 2
        self.K[1][2] = height / 2
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Apply mask on max distance allowed
        valid_mask = (depth_image.ravel() > 0) & (depth_image.ravel() < 3)
        M = np.column_stack((
            u.ravel()[valid_mask], 
            v.ravel()[valid_mask], 
            depth_image.ravel()[valid_mask], 
            np.ones(valid_mask.sum())
        ))

        # Convert camera frame -> world frame
        points = pixel_to_world(M, self.K, self.pose)
        # filtered_points = remove_outliers(points)
        # pillar_points = sspillarize_points_kmeans(points)
        np.save('gpoints.npy', points)
        # np.save('ppoints.npy', pillar_points)

        # Convert to PointCloud2
        header = msg.header
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1), # offset=0: The x coordinate starts at byte 0.
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1), # offset=4: The y coordinate starts at byte 4.
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1), # offset=8: The z coordinate starts at byte 8.
        ]

        # Publish Raw Point Cloud
        self.get_logger().info(f'pcl shape: {points[:, :3].shape}')
        pc_msg = pc2.create_cloud(header, fields, points[:, :3])
        pc_msg.header.frame_id = 'odom_frame'
        pc_msg.header.stamp = self.get_clock().now().to_msg()
        self.pc_pub.publish(pc_msg)

        # # Publish Denoised Pillar Point Cloud
        # self.get_logger().info(f'pcl shape: {pillar_points[:, :3].shape}')
        # d_pc_msg = pc2.create_cloud(header, fields, pillar_points[:, :3])
        # d_pc_msg.header.frame_id = 'odom_frame'
        # d_pc_msg.header.stamp = self.get_clock().now().to_msg()
        # self.denoised_pub.publish(d_pc_msg)

        # Measure Latency
        now = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9
        pcl_latency = now - start
        self.get_logger().info(f"Pcl Process Time: {pcl_latency:.6f} s" )
        

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
    
    '''
    def depth_callback(self, msg):
        """ Process depth image and publish waypoint for furthest valid point at drone height. """
        if self.K is None or self.pose['position'] is None:
            return  # Wait for valid camera intrinsics & pose

        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        height, width = depth_image.shape
        drone_height = self.pose['position'][2]

        max_depth = -1
        best_world_point = None

        # Iterate through pixels to find the furthest point close to drone height
        for v in range(height):
            for u in range(width):
                depth = depth_image[v, u]
                if depth > 0:  # Ignore invalid depth
                    world_coords = pixel_to_world(u, v, depth, self.K, self.pose)
                    if abs(world_coords[2] - drone_height) < 0.2:  # Height threshold
                        if depth > max_depth:  # Find the furthest valid point
                            max_depth = depth
                            best_world_point = world_coords

        if best_world_point is not None:
            self.publish_waypoint(best_world_point)

    def publish_waypoint(self, position):
        """ Publish a PoseStamped waypoint to /mavros/setpoint_position/local. """
        waypoint = PoseStamped()
        waypoint.header.stamp = self.get_clock().now().to_msg()
        waypoint.header.frame_id = "map"

        waypoint.pose.position.x = position[0]
        waypoint.pose.position.y = position[1]
        waypoint.pose.position.z = position[2]

        waypoint.pose.orientation.w = 1.0  # Neutral orientation

        self.waypoint_pub.publish(waypoint)
        self.get_logger().info(f"Published waypoint: {position}")
    '''

def main(args=None):
    rclpy.init(args=args)
    node = DepthToPointCloud()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()