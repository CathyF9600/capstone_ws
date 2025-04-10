import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2
import tf_transformations
from nav_msgs.msg import Odometry
import threading
import msvcrt  # For Windows. On Linux/Mac, use 'curses' or another library

# Constants
DISTANCE = 10.0
VOXEL_SIZE = 0.3
H, W = 200, 752
CAM_POSE = [0, 0, 0, 1, 0, 0, 0]  # Example: Quaternion [qx, qy, qz, qw] and translation [X_w, Y_w, Z_w]

# Function to transform camera frame to world frame
def transform_cam_to_world(P_c):
    qx, qy, qz, qw, X_w, Y_w, Z_w = CAM_POSE
    R_cam_to_drone = np.array([
        [ 0,  0, 1],  # X_d = -Z_c
        [-1,  0,  0],  # Y_d = -X_c
        [ 0, 1,  0],  # Z_d = -Y_c
    ])
    T_cam_to_drone = np.eye(4)
    T_cam_to_drone[:3, :3] = R_cam_to_drone

    R = tf_transformations.quaternion_matrix([qx, qy, qz, qw])[:3, :3]
    t = np.array([X_w, Y_w, Z_w])
    T_drone_to_world = np.block([
        [R, t.reshape(3, 1)], 
        [np.zeros((1, 3)), 1]
    ])

    T_cam_to_world = T_drone_to_world @ T_cam_to_drone

    if P_c.ndim == 1:
        P_c = P_c.reshape(1, 3)

    P_c_h = np.hstack([P_c, np.ones((P_c.shape[0], 1))])  # [N, 4]
    P_w_h = (T_cam_to_world @ P_c_h.T).T  # [N, 4]
    return P_w_h[:, :3].tolist()[0]  # Drop the homogeneous part

# RGBD Point Cloud Publisher Node
class RGBDPointCloudPublisher(Node):
    def __init__(self):
        super().__init__('rgbd_pointcloud_publisher')
        self.pose_sub = self.create_subscription(Odometry, '/camera/pose/sample', self.pose_callback, qos_profile_system_default)

        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/rgbd_data',
            self.listener_callback,
            10
        )
        
        # PCL PointCloud2 publisher
        self.pcl_pub = self.create_publisher(PointCloud2, '/rgbd_pointcloud', 10)

        self.rgbd_data = None
        self.received = False
        self.recording = True  # Flag to indicate whether we are recording

        # Start the keyboard listener in a separate thread
        self.listener_thread = threading.Thread(target=self.wait_for_enter_key)
        self.listener_thread.daemon = True
        self.listener_thread.start()

    def pose_callback(self, msg):
        # Extract pose information (camera pose in world frame)
        CAM_POSE[0:4] = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                         msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        CAM_POSE[4:7] = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]

    def listener_callback(self, msg):
        self.rgbd_data = np.array(msg.data, dtype=np.float32).reshape((H, W, 4))
        self.received = True
        self.get_logger().info("Received RGBD data.")

        if self.rgbd_data.any() and self.recording:
            self.process_and_publish_point_cloud()

    def process_and_publish_point_cloud(self):
        # Extract color and depth images
        rgbd_data = self.rgbd_data.copy()
        color_image = rgbd_data[..., :3]
        depth_image = np.clip(rgbd_data[..., 3], 0, DISTANCE)

        # Camera intrinsics
        fx = fy = 286.1167907714844
        cx, cy = W / 2, H / 2

        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        X = (xx - cx) * depth_image / fx
        Y = (yy - cy) * depth_image / fy
        Z = depth_image

        points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        colors = color_image.reshape(-1, 3) / 255.0

        # Transform points from camera frame to world frame
        points_world = np.array([transform_cam_to_world(p) for p in points])

        # Create Open3D point cloud and save
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Publish PointCloud2 message
        pcl_msg = self.create_pcl_msg(pcd)
        self.pcl_pub.publish(pcl_msg)

        # Save to .npy file in global frame
        np.save("point_cloud_global_frame.npy", points_world)

    def create_pcl_msg(self, pcd):
        # Convert Open3D point cloud to PointCloud2 message
        points = np.asarray(pcd.points)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"  # Adjust the frame_id if needed

        # Convert to PointCloud2
        pc_data = point_cloud2.create_cloud_xyz32(header, points)
        return pc_data

    def wait_for_enter_key(self):
        # Wait for the "Enter" key to stop recording
        while True:
            if msvcrt.kbhit():  # For Windows, use 'curses' for Linux/Mac
                key = msvcrt.getch()
                if key == b'\r':  # Enter key
                    self.recording = False
                    self.get_logger().info("Recording stopped by Enter key.")
                    break

def main(args=None):
    rclpy.init(args=args)
    node = RGBDPointCloudPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
