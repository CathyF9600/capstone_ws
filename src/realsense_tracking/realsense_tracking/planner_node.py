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
import queue

DISTANCE = 10.0
STEP = 0.5
VOXEL_SIZE = 0.08
COLOR_THRESHOLD = 0.3
MAX_DEPTH = 15
H, W = 200, 752

def build_voxel_index_map(voxels):
    return {tuple(voxel.grid_index): voxel for voxel in voxels}

def get_voxel_color_fast(voxel_map, v_idx):
    voxel = voxel_map.get(tuple(v_idx))
    return voxel.color[0] if voxel else None

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def vplot(path, vis):
    if len(path) < 2:
        print("Path too short to draw.")
        return
    points = [list(p) for p in path]
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    colors = [[0, 0, 1] for _ in lines]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)

class RGBDPathPlanner(Node):
    def __init__(self, vis, geometry_queue):
        super().__init__('rgbd_planner')
        self.vis = vis
        self.geometry_queue = geometry_queue
        self.rgbd_data = None
        self.received = False

        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/rgbd_data',
            self.listener_callback,
            10
        )
        self.lock = threading.Lock()

    def listener_callback(self, msg):
        with self.lock:
            geometries = self.process_data(msg)
            if geometries:
                self.geometry_queue.put(geometries)

    def process_data(self, msg):
        flat = np.array(msg.data, dtype=np.float32)
        self.rgbd_data = flat.reshape((H, W, 4))
        self.received = True
        self.get_logger().info("Received RGBD data.")
        if not self.rgbd_data.any():
            return

        rgbd_data = self.rgbd_data.copy()
        color_image = rgbd_data[..., :3]
        depth_image = np.clip(rgbd_data[..., 3], 0, DISTANCE)

        fx = fy = 286.1167907714844
        cx, cy = W / 2, H / 2

        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
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

        occupancy = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, VOXEL_SIZE)
        voxels = occupancy.get_voxels()
        voxel_map = build_voxel_index_map(voxels)

        start = np.array([0.0, 0.0, 0.0])
        gpos = np.array([2.0, 0.0, -5.0])

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.paint_uniform_color([0, 1, 0])
        sphere.translate(gpos)

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

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
                break

            if np.linalg.norm(current_pos - goal) < 0.1:
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

        line_path = None
        if waypoint:
            line_path = o3d.geometry.LineSet()
            #vplot(waypoint, self.vis)

        return [pcd, occupancy, sphere, axis, line_path]

def set_top_down_view(vis, zoom_level=0.2, rotation_angle=90):
    ctr = vis.get_view_control()
    
    # Set zoom level
    ctr.set_zoom(zoom_level)  # Adjust zoom level (lower value = more zoomed in)
    
    # Rotate to a top-down view (around Z-axis, typically 90 degrees)
    ctr.rotate(rotation_angle, 0)  # Rotate 90 degrees around the Z-axis to look down
    
def main(args=None):
    rclpy.init(args=args)

    vis = o3d.visualization.Visualizer()
    vis.create_window("RGBD Path Planner", 800, 600)
    geometry_queue = queue.Queue()

    node = RGBDPathPlanner(vis, geometry_queue)

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    try:
        while True:
            while not geometry_queue.empty():
                geometries = geometry_queue.get()
                vis.clear_geometries()
                for g in geometries:
                    vis.add_geometry(g)
            set_top_down_view(vis)
            vis.poll_events()
            vis.update_renderer()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        vis.destroy_window()
