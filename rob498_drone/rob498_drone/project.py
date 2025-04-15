#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
import cv2
from message_filters import ApproximateTimeSynchronizer, Subscriber
from your_planning_helpers import transform_cam_to_world, heuristic, pad, is_line_free, add_progress_point, build_voxel_index_map, get_voxel_color_fast

class PlannerNode(Node):
    def __init__(self):
        super().__init__('drone_comm')
        self.bridge = CvBridge()
        self.solution = []
        self.global_path = []

        self.rgb_sub = Subscriber(self, Image, '/rgb_image')
        self.depth_sub = Subscriber(self, Image, '/depth_image')
        self.pose_sub = Subscriber(self, Odometry, '/camera/pose/sample')
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.pose_sub],
            queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.callback)

        self.path_pub = self.create_publisher(Path, '/planned_path', 10)

        self.get_logger().info("PlannerNode initialized and waiting for data...")

    def callback(self, rgb_msg, depth_msg, pose_msg):
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            if depth.ndim == 3:
                depth = depth[:, :, 0]

            h, w = depth.shape
            rgb = cv2.resize(rgb, (w, h))
            K = np.array([[525.0, 0.0, w // 2],
                          [0.0, 525.0, h // 2],
                          [0.0, 0.0, 1.0]])

            # Generate point cloud in camera frame
            xs, ys = np.meshgrid(np.arange(w), np.arange(h))
            xs = xs.flatten()
            ys = ys.flatten()
            zs = depth.flatten() / 1000.0  # assuming mm -> meters
            valid = zs > 0
            xs = xs[valid]
            ys = ys[valid]
            zs = zs[valid]

            Xs = (xs - K[0, 2]) * zs / K[0, 0]
            Ys = (ys - K[1, 2]) * zs / K[1, 1]
            Zs = zs
            points_cam = np.stack((Xs, Ys, Zs), axis=1)

            # Get drone pose
            pose = [
                pose_msg.pose.pose.orientation.x,
                pose_msg.pose.pose.orientation.y,
                pose_msg.pose.pose.orientation.z,
                pose_msg.pose.pose.orientation.w,
                pose_msg.pose.pose.position.x,
                pose_msg.pose.pose.position.y,
                pose_msg.pose.pose.position.z,
            ]

            # Transform point cloud to world frame
            points_world = transform_cam_to_world(points_cam, pose)

            # Build voxel grid
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_world)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.2)
            voxel_map = build_voxel_index_map(voxel_grid.get_voxels())

            # Set start and goal (e.g., goal 2m ahead of drone)
            start = np.array([pose[4], pose[5], pose[6]])
            goal = start + np.array([-1.2, 0.0, -2.8])

            # Simple A* path planning
            open_set = []
            heapq.heappush(open_set, (0, 0, tuple(start)))
            came_from = {}
            cost_so_far = {tuple(start): 0}
            counter = count()

            found = False
            while open_set:
                _, _, current = heapq.heappop(open_set)
                if np.linalg.norm(np.array(current) - goal) < 0.2:
                    found = True
                    break
                for neighbor in pad(np.array(current)):
                    neighbor_t = tuple(neighbor.round(2))
                    if tuple(voxel_grid.get_voxel(neighbor)) not in voxel_map:
                        new_cost = cost_so_far[current] + np.linalg.norm(neighbor - np.array(current))
                        if neighbor_t not in cost_so_far or new_cost < cost_so_far[neighbor_t]:
                            cost_so_far[neighbor_t] = new_cost
                            priority = new_cost + heuristic(neighbor, goal)
                            heapq.heappush(open_set, (priority, next(counter), neighbor_t))
                            came_from[neighbor_t] = current

            # Reconstruct path
            if found:
                path = []
                current = tuple(goal.round(2))
                while current != tuple(start):
                    path.append(current)
                    current = came_from.get(current, tuple(start))
                path.append(tuple(start))
                path.reverse()

                self.publish_path(path, rgb_msg.header)
            else:
                self.get_logger().warn("Path not found!")

        except Exception as e:
            self.get_logger().error(f"Callback error: {e}")

    def publish_path(self, path, header):
        ros_path = Path()
        ros_path.header = Header()
        ros_path.header.stamp = header.stamp
        ros_path.header.frame_id = 'map'
        for pt in path:
            pose = PoseStamped()
            pose.header = ros_path.header
            pose.pose.position.x = pt[0]
            pose.pose.position.y = pt[1]
            pose.pose.position.z = pt[2]
            pose.pose.orientation.w = 1.0
            ros_path.poses.append(pose)
        self.path_pub.publish(ros_path)
        self.get_logger().info(f"Published path with {len(path)} points.")

def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
