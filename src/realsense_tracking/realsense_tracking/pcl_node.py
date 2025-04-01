#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
from pcl import PointCloud # PCL in Python
import image_geometry
from rclpy.qos import qos_profile_system_default

class PillarDetection(Node):
    def __init__(self):
        super().__init__('realsense_tracking')
        self.bridge = CvBridge()
        self.camera_model = None
        
        self.create_subscription(CameraInfo, '/camera/fisheye1/camera_info', self.camera_info_callback, qos_profile_system_default)
        self.create_subscription(Image, '/depth_image', self.depth_callback, qos_profile_system_default)
        self.cluster_pub = self.create_publisher(PointCloud2, '/pillar_clusters', qos_profile_system_default)

    def camera_info_callback(self, msg):
        self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_model.fromCameraInfo(msg)

    def depth_callback(self, msg):
        if self.camera_model is None:
            self.get_logger().warn("Waiting for camera info...")
            return
        
        # Convert depth image to numpy
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        height, width = depth_image.shape
        self.get_logger().info(f'depth shape: {height}x{width}')
        
        # Generate 3D points
        points = []
        for v in range(height):
            for u in range(width):
                z = depth_image[v, u] / 1000.0  # Convert mm to meters if needed
                if z == 0:
                    continue
                x, y, _ = self.camera_model.projectPixelTo3dRay((u, v))
                x, y = x * z, y * z
                points.append((x, y, z))
        
        if not points:
            self.get_logger().warn("No valid depth points")
            return
        
        # Convert to PCL point cloud
        pcl_cloud = pcl.PointCloud() # pc = pcl.PointCloud(np.array(points, dtype=np.float32))
        pcl_cloud.from_list(points)
        
        # Voxel Grid Downsampling
        voxel = pcl_cloud.make_voxel_grid_filter()
        voxel.set_leaf_size(0.05, 0.05, 0.05)
        pcl_cloud = voxel.filter()
        
        # Pass-through filter (remove ground)
        passthrough = pcl_cloud.make_passthrough_filter()
        passthrough.set_filter_field_name("z")
        passthrough.set_filter_limits(0.2, 2.0)
        pcl_cloud = passthrough.filter()
        
        # Euclidean Clustering
        tree = pcl_cloud.make_kdtree()
        ec = pcl_cloud.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(0.3)
        ec.set_MinClusterSize(50)
        ec.set_MaxClusterSize(10000)
        ec.set_SearchMethod(tree)
        cluster_indices = ec.Extract()
        
        # Convert clusters to PointCloud2
        cluster_cloud = pcl.PointCloud()
        for idx in cluster_indices:
            for i in idx.indices:
                cluster_cloud.push_back(pcl_cloud[i])
        
        ros_cluster_cloud = point_cloud2.create_cloud_xyz32(msg.header, cluster_cloud.to_array())
        self.cluster_pub.publish(ros_cluster_cloud)
        self.get_logger().info("Published clustered pillars")


def main():
    rclpy.init()
    node = PillarDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
