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
import heapq
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy, qos_profile_system_default
import tf_transformations
from itertools import count
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent

# Constants
DISTANCE = 10.0 
STEP = 0.2
VOXEL_SIZE = 0.1
COLOR_THRESHOLD = 0.4 # color
MAX_DEPTH = 100
PAD_DIST = 0.2
GLOBAL_SOLUTION = []

# Helper functions
# Function to transform camera frame to world frame using broadcasting
def transform_cam_to_world(P_c, pose): # [N, 4]
    qx, qy, qz, qw, X_w, Y_w, Z_w = pose
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

    P_c_h = np.hstack([P_c, np.ones((P_c.shape[0], 1))])  # [N, 4]
    P_w_h = (T_cam_to_world @ P_c_h.T).T  # [N, 4]
    return P_w_h[:, :3]  # Drop the homogeneous part


def add_progress_point(current_plan, global_path, full_goal, min_progress_distance=0.3):
    """
    Adds the first point from current_plan that is:
    - closer to the goal than the last point in global_path
    - AND sufficiently far from the last point (to avoid jittering).
    
    Returns the point added to global_path, or None if no point was added.
    """
    if not current_plan:
        return None

    if not global_path:
        first_pt = tuple(current_plan[0])
        global_path.append(first_pt)
        return current_plan[0]

    last_point = np.array(global_path[-1])
    goal = np.array(full_goal)
    last_dist_to_goal = np.linalg.norm(goal - last_point)

    for pt in current_plan:
        pt = np.array(pt)
        dist_to_goal = np.linalg.norm(goal - pt)
        dist_to_last = np.linalg.norm(pt - last_point)

        if dist_to_goal < last_dist_to_goal and dist_to_last > min_progress_distance:
            new_pt = tuple(pt)
            global_path.append(new_pt)
            return pt

    return None


def build_voxel_index_map(voxels):
    """
    Build a dictionary for fast lookup of voxels by their grid_index.
    """
    voxel_map = {}
    for voxel in voxels:
        voxel_map[tuple(voxel.grid_index)] = voxel  # Using tuple to make the index hashable
    return voxel_map


def get_voxel_color_fast(voxel_map, v_idx):
    """
    Returns the color of the voxel at the given index, using the pre-built voxel_map for O(1) lookup.
    """
    v_idx_tuple = tuple(v_idx)  # Ensure v_idx is hashable (tuple)
    voxel = voxel_map.get(v_idx_tuple)  # O(1) average-time lookup
    return voxel.color[0] if voxel else None


def heuristic(a, b):
    # Heuristic: Euclidean distance between 'a' and 'b'
    return np.linalg.norm(np.array(a) - np.array(b))


def pad(current_pos): # skip checking above and below (since obstacles are pillars)
    return [
        # Axis-aligned neighbors
        current_pos + np.array([PAD_DIST, 0, 0]),
        current_pos + np.array([-PAD_DIST, 0, 0]),
        current_pos + np.array([0, PAD_DIST, 0]),
        current_pos + np.array([0, -PAD_DIST, 0]),
        # Diagonal neighbors on xz-plane
        current_pos + np.array([PAD_DIST, PAD_DIST, 0]),
        current_pos + np.array([-PAD_DIST, PAD_DIST, 0]),
        current_pos + np.array([PAD_DIST, -PAD_DIST, 0]),
        current_pos + np.array([-PAD_DIST, -PAD_DIST, 0])
    ]


def is_line_free(p1, p2, occupancy, voxel_map, step=0.1):
    direction = p2 - p1
    distance = np.linalg.norm(direction)
    direction /= distance
    steps = int(distance / step)
    for i in range(1, steps + 1):
        point = p1 + i * step * direction
        v_idx = occupancy.get_voxel(point)
        if v_idx is not None:
            for dx in range(-int(PAD_DIST), int(PAD_DIST)+1):
                for dy in range(-int(PAD_DIST), int(PAD_DIST)+1):
                    for dz in range(-int(PAD_DIST), int(PAD_DIST)+1):
                        neighbor_idx = (v_idx[0]+dx, v_idx[1]+dy, v_idx[2]+dz)
                        if neighbor_idx in voxel_map:
                            color = get_voxel_color_fast(voxel_map, neighbor_idx)
                            if color and color < COLOR_THRESHOLD:
                                return False  # obstacle detected
    return True


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

        # self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.setpoint_publisher = self.create_publisher(PoseStamped, "/mavros/setpoint_position/local", qos_profile_system_default)

        # Variables
        self.current_pose = None
        self.goal = np.array([10.0, 10.0])  # Set your goal here
        self.global_path = []
        self.waiting_for_input = False
        self.next_waypoint = None

        # Visualization
        self.fig, self.ax = plt.subplots()
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.ion()
        
        self.get_logger().info("PlannerNode initialized and waiting for data...")

    def callback(self, rgb_msg, depth_msg, pose_msg):
        if self.waiting_for_input:
            return
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')  # HxWx3
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')  # HxW
            if depth_image.ndim == 3:
                depth_image = depth_image[:, :, 0]  # If it's 3D, collapse to 2D

            h, w = depth_image.shape
            color_image = cv2.resize(rgb_image, (w, h))
            depth_image = np.clip(depth_image, 0, DISTANCE)
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
            # Simulate planner result (replace with your planner call)
            next_wp = self.planner(color_image, depth_image, pose)
    
            if next_wp is not None:
                self.next_waypoint = next_wp
                self.waiting_for_input = True
                self.plot_state()
        
        except Exception as e:
            self.get_logger().error(f"Callback error: {e}")

    def planner(self, color_image, depth_image, pose):
            fx = fy = 286.1167907714844
            cx, cy = depth_image.shape[1] / 2, depth_image.shape[0] / 2

            xx, yy = np.meshgrid(np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0]))
            X = (xx - cx) * depth_image / fx
            Y = (yy - cy) * depth_image / fy
            Z = depth_image

            points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

            # Transform point cloud to world frame
            points_world = transform_cam_to_world(points, pose)

            # Build voxel grid
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_world)
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, VOXEL_SIZE)
            voxel_map = build_voxel_index_map(voxel_grid.get_voxels())

            # Set start and goal (e.g., goal 2m ahead of drone)
            # start = np.array([pose[4], pose[5], pose[6]])
            # goal = start + np.array([-1.2, 0.0, -2.8])

            # Short-Horizon Anytime A* path planning
            # START of A*
            waypoint = []
            # Plan waypoint based on gpos
            start = pose
            goal = self.goal
            tiebreaker = count()

            # Initialize A* search variables
            open_list = []
            heapq.heappush(open_list, (0 + heuristic(start, goal), 0, next(tiebreaker), start))  # f_score, g_score, position
            came_from = {}
            g_score = {tuple(start): 0}
            depth = 0
            # anytime A*
            found = False
            best_pos = None
            best_dist = float('inf')
            visited = set()

            while open_list:
                _, current_g_score, _, current_pos = heapq.heappop(open_list)
                visited.add(tuple(current_pos))
                dist_to_goal = np.linalg.norm(current_pos - goal)

                if depth >= MAX_DEPTH:
                    print("Max depth reached, stopping the pathfinding.")
                    break

                if np.linalg.norm(current_pos - goal) < 0.1:
                    print("Path found!")
                    found = True
                    break
                    
                # Track best position seen so far - anytime A*
                if dist_to_goal < best_dist:
                    best_dist = dist_to_goal
                    best_pos = current_pos
                
                # world frame planning - z is up, x cant be neg, y cant be pos (based on myhal room map)
                if tuple(current_pos) == tuple(start): # larger step to skip its own voxel
                    init_step = 2.5 * VOXEL_SIZE
                    neighbors = [
                        # Axis-aligned neighbors
                        current_pos + np.array([init_step, 0, 0]),
                        current_pos + np.array([0, -init_step, 0]),
                        current_pos + np.array([0, 0, init_step])
                    ]
                else:
                # Explore neighbors (simple 6-connected grid movement for 3D)
                    neighbors = [
                        # Axis-aligned neighbors
                        current_pos + np.array([STEP, 0, 0]),
                        current_pos + np.array([0, -STEP, 0]),
                        current_pos + np.array([0, 0, STEP]),
                        current_pos + np.array([0, 0, -STEP]),
                        # Diagonal neighbors on xz-plane
                        current_pos + np.array([STEP, 0, STEP]),
                        current_pos + np.array([0, -STEP, STEP]),
                        current_pos + np.array([STEP, 0, -STEP]),
                        current_pos + np.array([0, -STEP, -STEP])
                    ]

                for neighbor in neighbors:
                    if tuple(neighbor) in visited:
                        continue
                    visited.add(tuple(neighbor))
                    v_idx = voxel_grid.get_voxel(neighbor) 
                    if v_idx is not None:  # Skip occupied voxels
                        # check cell
                        color = get_voxel_color_fast(voxel_map, v_idx)
                        if color:
                            # print(f'obstacle found at {color:.2f}')
                            if color > COLOR_THRESHOLD: # voxel intensity threhold
                                # print(f'obstacle found at {color:.2f}')
                                continue
                        # check cell padded surroundings
                        is_near_obstacle = False
                        for padded_neighbor in pad(neighbor):
                            neighbor_idx = voxel_grid.get_voxel(padded_neighbor)
                            # print('padded_neighbor', neighbor, padded_neighbor)
                            if neighbor_idx is not None:
                                color = get_voxel_color_fast(voxel_map, neighbor_idx)
                                if color:
                                    # print(f'obstacle found at {color:.2f}')
                                    if color > COLOR_THRESHOLD:
                                        is_near_obstacle = True
                                        cached_voxel = o3d.geometry.Voxel(neighbor_idx, (np.array([color,color,color])))
                                        voxel_map[tuple(neighbor_idx)] = cached_voxel # caching
                                        break
                        if is_near_obstacle:
                            # print(f'obstacle found near {neighbor}')
                            continue  # Skip this neighbor – treated as inflated obstacle

                    tentative_g_score = current_g_score + np.linalg.norm(neighbor - current_pos)

                    if tuple(neighbor) not in g_score or tentative_g_score < g_score[tuple(neighbor)]:
                        g_score[tuple(neighbor)] = tentative_g_score
                        f_score = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_score, tentative_g_score, next(tiebreaker), neighbor)) #https://stackoverflow.com/questions/39504333/python-heapq-heappush-the-truth-value-of-an-array-with-more-than-one-element-is
                        came_from[tuple(neighbor)] = current_pos
                depth += 1

            # Use found path or fallback
            if found:
                target_pos = current_pos
                print("Reached goal!")
            else:
                target_pos = best_pos
                print(f"Depth limit reached — using best-effort path (dist {best_dist:.2f})")
            waypoint = [target_pos]
            while tuple(waypoint[-1]) in came_from:
                waypoint.append(came_from[tuple(waypoint[-1])])

            # Waypoint visualization
            waypoint.reverse()
            print("Original Path:", len(waypoint))

            # Path pruning
            pruned_path = [waypoint[0]]
            i = 0
            while i < len(waypoint) - 1:
                j = len(waypoint) - 1
                while j > i + 1:
                    if is_line_free(waypoint[i], waypoint[j], voxel_grid, voxel_map):
                        break
                    j -= 1
                pruned_path.append(waypoint[j])
                i = j
            print("Pruned Path:", pruned_path)

            new_points = add_progress_point(waypoint, self.global_path, full_goal=self.goal)
            if new_points:
                print('new_points', new_points)
                return new_points
                
    def on_key(self, event: KeyEvent):
        if event.key == 'enter' and self.waiting_for_input:
            self.global_path.append(tuple(self.next_waypoint))
            self.waiting_for_input = False
            self.plot_state()
        elif event.key == 'escape':
            rclpy.shutdown()
            
    def plot_state(self):
        self.ax.clear()
        self.ax.plot(self.goal[0], self.goal[1], 'ro', label='Goal')
        if self.current_pose is not None:
            self.ax.plot(self.current_pose[0], self.current_pose[1], 'bo', label='Current')
        if self.global_path:
            path = np.array(self.global_path)
            self.ax.plot(path[:, 0], path[:, 1], 'g--', label='Global Path')
        if self.next_waypoint is not None:
            self.ax.plot(self.next_waypoint[0], self.next_waypoint[1], 'kx', label='Next Waypoint')

        self.ax.set_title("Press ENTER to accept waypoint, ESC to quit")
        self.ax.set_xlim(-1, 15)
        self.ax.set_ylim(-1, 15)
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
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
    planner = PlannerNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(planner, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
