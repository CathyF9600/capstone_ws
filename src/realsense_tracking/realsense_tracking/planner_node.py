import numpy as np
import open3d as o3d
import cv2
import heapq
from itertools import count
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

DISTANCE = 10.0
STEP = 0.5
VOXEL_SIZE = 0.08
COLOR_THRESHOLD = 0.3  # voxel intensity
MAX_DEPTH = 15
H, W = 200, 752

def build_voxel_index_map(voxels):
    voxel_map = {}
    for voxel in voxels:
        voxel_map[tuple(voxel.grid_index)] = voxel
    return voxel_map

def get_voxel_color_fast(voxel_map, v_idx):
    voxel = voxel_map.get(tuple(v_idx))
    return voxel.color[0] if voxel else None

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def vplot(path, vis):
    color = [0, 0, 1]
    if len(path) < 2:
        print("Path too short to draw.")
        return
    points = [list(p) for p in path]
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    colors = [color for _ in lines]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)

class RGBDPathPlanner(Node):
    def __init__(self):
        super().__init__('rgbd_planner')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/rgbd_data',
            self.listener_callback,
            10
        )
        self.lock = threading.Lock()
        self.received = False
        self.rgbd_data = None
        self.vis = None
        self.data_ready = False
        self.visualization_thread = threading.Thread(target=self.visualizer_loop, daemon=True)
        self.visualization_thread.start()

    def listener_callback(self, msg):
        with self.lock:
            # Assuming msg.data is flattened (H x W x 4)
            flat = np.array(msg.data, dtype=np.float32)
            size = int(flat.size // 4)
            # H, W =   # assumes square images
            self.rgbd_data = flat.reshape((H, W, 4))
            self.received = True
            self.data_ready = True
            self.get_logger().info("Received RGBD data.")
        
    def process_and_visualize(self):
        if not self.rgbd_data:
            return
        rgbd_data = self.rgbd_data.copy()
        color_image = rgbd_data[..., :3]
        depth_image = np.clip(rgbd_data[..., 3], 0, DISTANCE)

        fx = fy = 286.1167907714844
        cx, cy = depth_image.shape[1] / 2, depth_image.shape[0] / 2

        xx, yy = np.meshgrid(np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0]))
        X = (xx - cx) * depth_image / fx
        Y = (yy - cy) * depth_image / fy
        Z = depth_image

        points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        colors = color_image.reshape(-1, 3) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])

        if self.vis is None:
            self.vis = o3d.visualization.VisualizerWithKeyCallback()
            self.vis.create_window(window_name="RGB-D Path Planning", width=800, height=600)
        
        self.vis.clear_geometries()
        self.vis.add_geometry(pcd)

        occupancy = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, VOXEL_SIZE)
        voxels = occupancy.get_voxels()
        voxel_map = build_voxel_index_map(voxels)
        self.vis.add_geometry(occupancy)

        start = np.array([0.0, 0.0, 0.0])
        gpos = np.array([2.0, 0.0, -5.0])
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.paint_uniform_color([0, 1, 0])
        sphere.translate(gpos)
        self.vis.add_geometry(sphere)

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0, 0, 0]))
        self.vis.add_geometry(axis)

        waypoint = []
        goal = gpos
        tiebreaker = count()
        open_list = []
        heapq.heappush(open_list, (0 + heuristic(start, goal), 0, next(tiebreaker), start))
        came_from = {}
        g_score = {tuple(start): 0}
        depth = 0
        last_valid_pos = None

        while open_list:
            _, current_g_score, _, current_pos = heapq.heappop(open_list)

            if depth >= MAX_DEPTH:
                print("Max depth reached.")
                break

            if np.linalg.norm(current_pos - goal) < 0.1:
                print("Goal reached.")
                break

            neighbors = [
                current_pos + np.array([STEP, 0, 0]),
                current_pos + np.array([-STEP, 0, 0]),
                current_pos + np.array([0, 0, STEP]),
                current_pos + np.array([0, 0, -STEP]),
                current_pos + np.array([STEP, 0, STEP]),
                current_pos + np.array([-STEP, 0, STEP]),
                current_pos + np.array([STEP, 0, -STEP]),
                current_pos + np.array([-STEP, 0, -STEP])
            ]

            for neighbor in neighbors:
                v_idx = occupancy.get_voxel(neighbor)
                if v_idx is not None:
                    color = get_voxel_color_fast(voxel_map, v_idx)
                    if color and color > COLOR_THRESHOLD:
                        continue
                else:
                    continue

                tentative_g_score = current_g_score + np.linalg.norm(neighbor - current_pos)
                if tuple(neighbor) not in g_score or tentative_g_score < g_score[tuple(neighbor)]:
                    g_score[tuple(neighbor)] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score, tentative_g_score, next(tiebreaker), neighbor))
                    came_from[tuple(neighbor)] = current_pos
                    last_valid_pos = neighbor
            depth += 1

        current_pos = last_valid_pos
        while tuple(current_pos) in came_from:
            waypoint.append(current_pos)
            current_pos = came_from[tuple(current_pos)]
        waypoint.append(start)
        waypoint.reverse()

        if waypoint:
            print("Path:", waypoint)
            vplot(waypoint, self.vis)

    def visualizer_loop(self):
        while rclpy.ok():
            if self.data_ready:
                with self.lock:
                    self.process_and_visualize()
                self.data_ready = False
            if self.vis:
                self.vis.poll_events()
                self.vis.update_renderer()

def main(args=None):
    rclpy.init(args=args)
    node = RGBDPathPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
