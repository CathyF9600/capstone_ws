import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import open3d as o3d
import heapq
from itertools import count
from cv_bridge import CvBridge
import time

DISTANCE = 10.0
STEP = 0.5
VOXEL_SIZE = 0.08
COLOR_THRESHOLD = 0.3  # color
MAX_DEPTH = 15
REFRESH_TIME = 0.5

class PathPlannerGUI(Node):
    def __init__(self):
        super().__init__('realsense_tracking')
        self.bridge = CvBridge()

        # Initialize the Open3D visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="RGB-D Point Cloud", width=800, height=600)

        # Subscribe to the depth and color image topics
        self.create_subscription(
            Image,
            '/depth_image',
            self.depth_image_callback,
            10
        )
        self.create_subscription(
            Image,
            '/color_image',
            self.color_image_callback,
            10
        )

        self.timer = self.create_timer(1.0, self.timer_callback)  # Update visualization every 1 second

        # To store the latest depth and color images
        self.latest_depth_image = None
        self.latest_color_image = None
        self.latest_time = time.time()

    def depth_image_callback(self, msg):
        # Convert the depth image from ROS to OpenCV format
        depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")  # 32-bit float depth image
        self.latest_depth_image = depth_image

    def color_image_callback(self, msg):
        # Convert the color image from ROS to OpenCV format
        color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")  # 8-bit color image (BGR)
        self.latest_color_image = color_image

    def timer_callback(self):
        # Check if both depth and color images are available and if enough time has passed
        if self.latest_depth_image is not None and self.latest_color_image is not None and (time.time() - self.latest_time) >= REFRESH_TIME:
            self.latest_time = time.time()

            # Process the depth and color images and update the visualization
            self.process_images(self.latest_depth_image, self.latest_color_image)
    
    def process_images(self, depth_image, color_image):
        # Clip depth image to max distance and convert to numpy array
        depth_image = np.clip(depth_image, 0, DISTANCE)

        # Camera intrinsic parameters (adjust if necessary)
        fx = fy = 286.1167907714844
        cx, cy = depth_image.shape[1] / 2, depth_image.shape[0] / 2

        xx, yy = np.meshgrid(np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0]))
        X = (xx - cx) * depth_image / fx
        Y = (yy - cy) * depth_image / fy
        Z = depth_image

        points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        # Normalize the color image to [0, 1]
        colors = color_image.reshape(-1, 3) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Open3D convention: Flip the point cloud to match ROS frame
        pcd.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])

        # Add point cloud to visualizer
        self.vis.clear_geometries()
        self.vis.add_geometry(pcd)

        # Update the voxel grid visualization (you may want to update this logic)
        occupancy = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, VOXEL_SIZE)
        self.vis.add_geometry(occupancy)

        # Finalize visualization update
        self.vis.update_geometry(pcd)
        self.vis.update_geometry(occupancy)
        self.vis.poll_events()
        self.vis.update_renderer()

    def run(self):
        # Start the visualization and spin ROS 2 node
        rclpy.spin(self)
        self.vis.destroy_window()

def main(args=None):
    rclpy.init(args=args)
    planner = PathPlanner()
    planner.run()

if __name__ == '__main__':
    main()
