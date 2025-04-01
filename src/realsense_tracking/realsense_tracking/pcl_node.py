import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import TransformStamped
import tf2_ros
from rclpy.qos import qos_profile_system_default
import open3d as o3d
from sklearn.cluster import DBSCAN

class PillarDetection(Node):
    def __init__(self):
        super().__init__('pillar_detection')
        self.bridge = CvBridge()

        self.fx, self.fy, self.cx, self.cy, self.baseline = None, None, None, None, None

        self.create_subscription(CameraInfo, '/camera/fisheye1/camera_info', self.camera_info_callback, 10)
        self.create_subscription(Image, '/depth_image', self.depth_callback, 10)

        self.pc_pub = self.create_publisher(PointCloud2, '/filtered_point_cloud', 10)

    def camera_info_callback(self, msg):
        """Retrieve camera intrinsics from CameraInfo."""
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.baseline = 0.064  # Stereo baseline (adjust as needed)

    def depth_callback(self, msg):
        """Convert depth image to clustered point cloud."""
        if None in (self.fx, self.fy, self.cx, self.cy, self.baseline):
            return

        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        height, width = depth_image.shape

        points = []
        for v in range(0, height, 4):  # Downsample by taking every 4th pixel
            for u in range(0, width, 4):
                disparity = depth_image[v, u]
                if disparity > 1.0:  # Ignore invalid depths
                    Z = (self.fx * self.baseline) / disparity
                    X = (u - self.cx) * Z / self.fx
                    Y = (v - self.cy) * Z / self.fy
                    points.append((X, Y, Z))

        if len(points) == 0:
            return

        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))

        # DBSCAN Clustering
        labels = np.array(pcd.cluster_dbscan(eps=0.2, min_points=10, print_progress=False))

        # Extract only larger clusters (potential pillars)
        unique_labels = set(labels)
        clustered_points = []
        for label in unique_labels:
            if label == -1:  # Ignore noise
                continue
            cluster = np.array(points)[labels == label]
            if len(cluster) > 20:  # Keep only significant clusters
                clustered_points.extend(cluster)

        if len(clustered_points) == 0:
            return

        # Convert back to PointCloud2 message
        header = msg.header
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        pc_msg = pc2.create_cloud(header, fields, clustered_points)
        self.pc_pub.publish(pc_msg)

def main():
    rclpy.init()
    node = PillarDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
