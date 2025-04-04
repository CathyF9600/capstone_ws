# from sklearn.cluster import MiniBatchKMeans
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid
import sensor_msgs_py.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros
from rclpy.qos import qos_profile_system_default
import tf_transformations
# from octomap_msgs.msg import Octomap
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from std_msgs.msg import Header


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


def create_occupancy_grid(occupied_voxels, voxel_size, height=0.5, grid_size=(100, 100)):
    """
    Creates a 2D occupancy grid for a given height.

    Args:
        occupied_voxels (np.ndarray): M x 3 array of occupied voxel coordinates.
        voxel_size (float): Size of each voxel (meters).
        height (float): The world height at which to extract the layer.
        grid_size (tuple): Fixed (width, height) of the occupancy grid.

    Returns:
        np.ndarray: 2D binary occupancy grid (grid_size_x, grid_size_y).
    """
    # Convert height to voxel index
    z_index = int(np.floor(height / voxel_size))

    # Extract the layer at this height
    layer_voxels = occupied_voxels[occupied_voxels[:, 2] == z_index]

    if layer_voxels.shape[0] == 0:
        return np.zeros(grid_size, dtype=np.uint8)  # Empty grid

    # Get min x, y to shift coordinates into grid space
    min_x, min_y = layer_voxels[:, 0].min(), layer_voxels[:, 1].min()

    # Shift coordinates so that (min_x, min_y) becomes (0,0)
    shifted_x = layer_voxels[:, 0] - min_x
    shifted_y = layer_voxels[:, 1] - min_y

    # Create an empty grid
    occupancy_grid = np.zeros(grid_size, dtype=np.uint8)

    # Clip to ensure indices stay within fixed grid bounds
    valid_x = np.clip(shifted_x, 0, grid_size[0] - 1)
    valid_y = np.clip(shifted_y, 0, grid_size[1] - 1)

    # Mark occupied cells
    occupancy_grid[valid_x, valid_y] = 1
    return occupancy_grid


# Convert to displayable grayscale image
def convert_occupancy_to_image(grid):
    image = np.zeros_like(grid, dtype=np.uint8)
    image[grid == -1] = 127     # Unknown -> Gray
    image[grid == 0] = 255      # Free -> White
    image[grid == 100] = 0      # Occupied -> Black
    return image


def voxel_occupancy_map(points, voxel_size=0.25, threshold=5):
    """
    Converts point cloud into a 3D occupancy grid using voxelization.
    
    Args:
        points (np.ndarray): N x 3 array of 3D points.
        voxel_size (float): Size of each voxel (in meters).
        threshold (int): Minimum number of points to mark a voxel as occupied.
    
    Returns:
        occupied_voxels (np.ndarray): M x 3 array of voxel coordinates that are occupied.
    """
    # Discretize points into voxel grid
    voxel_indices = np.floor(points / voxel_size).astype(int)
    # Count number of points per voxel
    unique_voxels, counts = np.unique(voxel_indices, axis=0, return_counts=True)
    # Filter voxels that exceed the threshold
    occupied_voxels = unique_voxels[counts >= threshold]
    return occupied_voxels


class DepthToPointCloud(Node):
    def __init__(self, resolution=0.2):
        super().__init__('realsense_tracking')
        # self.resolution = resolution
        # self.octree = octomap.OcTree(resolution)

        self.bridge = CvBridge()
        self.pose = {'position':None, 'orientation':None}
        # Camera parameters (to be updated from CameraInfo)
        self.fx, self.fy, self.cx, self.cy, self.baseline = None, None, None, None, None
        self.K, self.P = None, None
        # Subscribers
        self.create_subscription(CameraInfo, '/camera/fisheye1/camera_info', self.camera_info_callback, qos_profile_system_default)
        self.create_subscription(Image, '/depth_image', self.depth_callback, qos_profile_system_default)
        self.pose_sub = self.create_subscription(Odometry, '/camera/pose/sample', self.pose_callback, qos_profile_system_default)
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
        # self.octomap_pub = self.create_publisher(Octomap, "/octomap_binary", qos_profile_system_default)
        self.occu_pub = self.create_publisher(OccupancyGrid, '/occupancy_grid', qos_profile_system_default)

        # TF2 Buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.resolution = 0.2
        
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


    def visualize_occupancy_grid(self, grid, min_indices):
        """
        Visualizes a 2D occupancy grid using Matplotlib.
        
        Args:
            grid (np.ndarray): 2D binary occupancy grid (0 or 1 values).
            min_indices (tuple): The minimum (x, y) voxel indices used for adjustment.
        """
        # Extract obstacle coordinates (where grid value is 1)
        obstacle_coords = np.argwhere(grid == 1)
        
        # Adjust coordinates to account for the original indices (add min_indices back)
        obstacle_coords_adjusted = obstacle_coords + min_indices[:2]  # Only adjust x, y

        # Create a 2D plot
        fig, ax = plt.subplots()

        # Plot obstacles as red dots on the grid
        ax.scatter(obstacle_coords_adjusted[:, 1], obstacle_coords_adjusted[:, 0], c='red', marker='o')

        # Set labels and title
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_title('2D Occupancy Grid Visualization')

        # Show gridlines for clarity
        ax.grid(True)

        # Show the plot
        plt.show()


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


    def publish_grid(self, grid, min_indices):
        """Publishes the occupancy grid to ROS 2."""
        grid = grid.T
        if grid is None or min_indices is None:
            self.get_logger().warn("No grid data available yet.")
            return

        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id ='odom_frame'# self.frame_id

        msg.info.resolution = self.resolution
        msg.info.width = grid.shape[1]
        msg.info.height = grid.shape[0]
        
        msg.info.origin.position.x = min_indices[1] * self.resolution
        msg.info.origin.position.y = min_indices[0] * self.resolution
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        msg.data = [100 if cell == 1 else 0 for cell in grid.flatten()]

        self.occu_pub.publish(msg)
        self.get_logger().info("Published updated OccupancyGrid.")


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
        valid_mask = (depth_image.ravel() > 0) & (depth_image.ravel() < 10)
        M = np.column_stack((
            u.ravel()[valid_mask], 
            v.ravel()[valid_mask], 
            depth_image.ravel()[valid_mask], 
            np.ones(valid_mask.sum())
        ))

        # Convert camera frame -> world frame
        points = pixel_to_world(M, self.K, self.pose)[:, :3]
        # filtered_points = remove_outliers(points)
        # pillar_points = pillarize_points_kmeans(points)

        # np.save('gpoints.npy', points)
        # np.save('ppoints.npy', pillar_points)

        # # Convert to PointCloud2
        header = msg.header
        header = msg.header
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1), # offset=0: The x coordinate starts at byte 0.
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1), # offset=4: The y coordinate starts at byte 4.
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1), # offset=8: The z coordinate starts at byte 8.
        ]

        # # Publish Raw Point Cloud
        self.get_logger().info(f'pcl shape: {points.shape}')
        pc_msg = pc2.create_cloud(header, fields, points[:, :3])
        pc_msg.header.frame_id = 'odom_frame'
        pc_msg.header.stamp = self.get_clock().now().to_msg()
        self.pc_pub.publish(pc_msg)
        pc_msg = pc2.create_cloud(header, fields, points[:, :3])
        pc_msg.header.frame_id = 'odom_frame'
        pc_msg.header.stamp = self.get_clock().now().to_msg()
        self.pc_pub.publish(pc_msg)

        # # Publish Denoised Pillar Point Cloud
        # self.get_logger().info(f'pcl shape: {pillar_points[:, :3].shape}')
        # d_pc_msg = pc2.create_cloud(header, fields, pillar_points[:, :3])
        # d_pc_msg.header.frame_id = 'odom_frame'
        # d_pc_msg.header.stamp = self.get_clock().now().to_msg()
        # self.denoised_pub(d_pc_msg)

        # Compute occupied voxels
        # Set voxelization parameters
        voxel_size = self.resolution   # Each voxel is 0.5m x 0.5m x 0.5m
        threshold = 10     # A voxel is occupied if it has at least 10 points
        occupied_voxels = voxel_occupancy_map(points, voxel_size, threshold) # M x 3 array of occupied voxel coordinates
        # Find the voxel index corresponding to height 0.5m
        z_index = int(np.floor(0.5 / voxel_size))
        # Filter voxels at this height
        layer_voxels = occupied_voxels[occupied_voxels[:, 2] == z_index]
        self.get_logger().info(f"Number of occupied voxels at height 0.5m: {layer_voxels.shape[0]}")

        # Get 2D Binary occupancy grid 
        grid_size = (50, 50)  # Fixed 2D grid size
        height = 0.5  # Get occupancy grid at 0.5m height
        occupancy_grid = create_occupancy_grid(occupied_voxels, voxel_size=self.resolution, height=height, grid_size=grid_size)
        print(f"Occupancy grid at {height}m height:\n", occupancy_grid.shape)
        min_indices = (0, 0)  # No adjustment, but could be different depending on your data
        # self.publish_grid(occupancy_grid, min_indices)

        # Display occupancy_grid
        image = convert_occupancy_to_image(occupancy_grid)
        resized = cv2.resize(image, (300, 300), interpolation=cv2.INTER_NEAREST)

        # Show image
        cv2.imshow("Occupancy Grid Map", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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