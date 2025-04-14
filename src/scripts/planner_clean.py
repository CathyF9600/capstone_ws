import numpy as np
import open3d as o3d
import cv2
import heapq
from itertools import count

DISTANCE = 10.0
STEP = 0.2
VOXEL_SIZE = 0.1
COLOR_THRESHOLD = 0.1 # color
MAX_DEPTH = 50
PAD_DIST = 0.2

solution = [[0., 0., 0.], [ 0. ,  0. , -0.5], [-0.2,  0. , -0.7], [-0.4,  0. , -0.9], [-0.6,  0. , -0.9], [-0.8,  0. , -1.1], [-0.8,  0. , -1.3], [-0.8,  0. , -1.5], [-1. ,  0. , -1.7], [-1. ,  0. , -1.9], [-1.2,  0. , -2.1], [-1.2,  0. , -2.3], [-1.2,  0. , -2.5], [-1.2,  0. , -2.7], [-1.4,  0. , -2.9]]

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


def save_screenshot(vis):
    filename = "screenshot.png"
    vis.capture_screen_image(filename)
    print(f"Screenshot saved to {filename}")
    return False


def vplot(path, vis): # vector plot
    color = [0, 0, 1]
    if len(path) < 2:
        print("Path too short to draw.")
        return
    # Convert to Open3D-compatible format
    points = [list(p) for p in path]
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    # Create line set object
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    # Set color of each line
    colors = [color for _ in lines]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # Add to visualizer
    vis.add_geometry(line_set)


def pplot(vis, gpos, color='R'): # point plot
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)  # Larger sphere
    sphere.paint_uniform_color([1, 0, 0])  # Green color for the goal position
    sphere.translate(gpos)
    vis.add_geometry(sphere)

def pad(current_pos):
    return [
        # Axis-aligned neighbors
        current_pos + np.array([PAD_DIST, 0, 0]),
        current_pos + np.array([-PAD_DIST, 0, 0]),
        current_pos + np.array([0, 0, PAD_DIST]),
        current_pos + np.array([0, 0, -PAD_DIST]),
        # Diagonal neighbors on xz-plane
        current_pos + np.array([PAD_DIST, 0, PAD_DIST]),
        current_pos + np.array([-PAD_DIST, 0, PAD_DIST]),
        current_pos + np.array([PAD_DIST, 0, -PAD_DIST]),
        current_pos + np.array([-PAD_DIST, 0, -PAD_DIST])
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
                            if get_voxel_color_fast(voxel_map, neighbor_idx)[0] < COLOR_THRESHOLD:
                                return False  # obstacle detected
    return True

def plan_and_show_waypoint(fp, start=np.array([0.0, 0.0, 0.0]),gpos=np.array([2.0, 0.0, -5.0]), depth_threshold=3.0, occupancy_threshold=10):
    rgbd_data = np.load(fp, allow_pickle=True)
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

    # Open3D convention
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

    # Initialize Open3D visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="RGB-D Point Cloud", width=800, height=600)

    # Add point cloud to visualizer
    vis.add_geometry(pcd)

    # Occupancy map (Voxel grid)
    occupancy = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, VOXEL_SIZE)
    voxels = occupancy.get_voxels() # computationally expensive but i have no choice
    voxel_map = build_voxel_index_map(voxels=voxels)
    vis.add_geometry(occupancy)

    # Plot gpos as a bigger sphere (goal position)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)  # Larger sphere
    sphere.paint_uniform_color([0, 1, 0])  # Green color for the goal position
    sphere.translate(gpos)
    vis.add_geometry(sphere)

    # Add global coordinate axes at the origin (for XYZ axes)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0, 0, 0]))
    vis.add_geometry(axis)

    # START of A*
    waypoint = []
    # Plan waypoint based on gpos
    goal = gpos
    tiebreaker = count()

    # Initialize A* search variables
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, goal), 0, next(tiebreaker), start))  # f_score, g_score, position
    came_from = {}
    g_score = {tuple(start): 0}
    depth = 0
    last_valid_pos = None  # Variable to track the last valid position

    while open_list:
        _, current_g_score, _, current_pos = heapq.heappop(open_list)

        if depth >= MAX_DEPTH:
            print("Max depth reached, stopping the pathfinding.")
            break

        if np.linalg.norm(current_pos - goal) < 0.1:
            print("Path found!")
            break
        
        if tuple(current_pos) == tuple(start): # larger step to skip its own voxel
            init_step = 2.5 * VOXEL_SIZE
            neighbors = [
                # Axis-aligned neighbors
                # current_pos + np.array([init_step, 0, 0]),
                # current_pos + np.array([-init_step, 0, 0]),
                # current_pos + np.array([0, 0, init_step]),
                current_pos + np.array([0, 0, -0.5]),
                # # Diagonal neighbors on xz-plane
                # current_pos + np.array([init_step, 0, init_step]),
                # current_pos + np.array([-init_step, 0, init_step]),
                # current_pos + np.array([init_step, 0, -init_step]),
                # current_pos + np.array([-init_step, 0, -init_step])
            ]
        else:
        # Explore neighbors (simple 6-connected grid movement for 3D)
            neighbors = [
                # Axis-aligned neighbors
                current_pos + np.array([STEP, 0, 0]),
                current_pos + np.array([-STEP, 0, 0]),
                current_pos + np.array([0, 0, STEP]),
                current_pos + np.array([0, 0, -STEP]),
                # Diagonal neighbors on xz-plane
                current_pos + np.array([STEP, 0, STEP]),
                current_pos + np.array([-STEP, 0, STEP]),
                current_pos + np.array([STEP, 0, -STEP]),
                current_pos + np.array([-STEP, 0, -STEP])
            ]

        for neighbor in neighbors:
            # pplot(vis, neighbor, color='R')
            # print('n', neighbor, goal)
            v_idx = occupancy.get_voxel(neighbor) 
            if v_idx is not None:  # Skip occupied voxels
                # check cell
                color = get_voxel_color_fast(voxel_map, v_idx)
                if color:
                    print(f'obstacle found at {color:.2f}')
                    if color > COLOR_THRESHOLD: # voxel intensity threhold
                        print(f'obstacle found at {color:.2f}')
                        continue
                # check cell padded surroundings
                is_near_obstacle = False
                for padded_neighbor in pad(neighbor):
                    neighbor_idx = occupancy.get_voxel(padded_neighbor)
                    # print('padded_neighbor', neighbor, padded_neighbor)
                    if neighbor_idx is not None:
                        color = get_voxel_color_fast(voxel_map, neighbor_idx)
                        if color:
                            print(f'obstacle found at {color:.2f}')
                            if color > COLOR_THRESHOLD:
                                is_near_obstacle = True
                                cached_voxel = o3d.geometry.Voxel(neighbor_idx, (np.array([color,color,color])))
                                voxel_map[tuple(neighbor_idx)] = cached_voxel # caching
                                break
                if is_near_obstacle:
                    print(f'obstacle found near {neighbor}')
                    continue  # Skip this neighbor – treated as inflated obstacle
            else:
                print('v_idx is None!!')
            vplot(solution, vis)
            # pplot(vis, neighbor, color='R')

            tentative_g_score = current_g_score + np.linalg.norm(neighbor - current_pos)

            if tuple(neighbor) not in g_score or tentative_g_score < g_score[tuple(neighbor)]:
                g_score[tuple(neighbor)] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score, tentative_g_score, next(tiebreaker), neighbor)) #https://stackoverflow.com/questions/39504333/python-heapq-heappush-the-truth-value-of-an-array-with-more-than-one-element-is
                came_from[tuple(neighbor)] = current_pos
                last_valid_pos = neighbor
        depth += 1

    # Backtrack to create the path
    current_pos = last_valid_pos
    while tuple(current_pos) in came_from:
        waypoint.append(current_pos)
        current_pos = came_from[tuple(current_pos)]
    waypoint.append(start)

    # Waypoint visualization
    if waypoint is not None:
        waypoint.reverse()
        print("Original Path:", len(waypoint))
    
        # Path pruning
        pruned_path = [waypoint[0]]
        i = 0
        while i < len(waypoint) - 1:
            j = len(waypoint) - 1
            while j > i + 1:
                if is_line_free(waypoint[i], waypoint[j], occupancy, voxel_map):
                    break
                j -= 1
            pruned_path.append(waypoint[j])
            i = j
    
        print("Pruned Path:", len(pruned_path))
        vplot(pruned_path, vis)
    else:
        print('Not found!')


    # Register key to save screenshot
    vis.register_key_callback(ord("S"), save_screenshot)

    # Finalize visualization
    vis.run()
    vis.destroy_window()

# # Run with default gpos
# plan_and_show_waypoint("/Users/yuchunfeng/Downloads/rgbd.npy")

import os
import time

X = 5

def run_on_folder(folder_path, start=np.array([0.0, 0.0, 0.0]), gpos=np.array([-2.0, 0.0, -5.0])):
    npy_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])
    if not npy_files:
        print("No .npy files found in the folder.")
        return
    print('npy_files', len(npy_files))
    # input()
    # fname = npy_files[X]
    for fname in npy_files:
        full_path = os.path.join(folder_path, fname)
        print(f"\nShowing: {full_path}")
        try:
            plan_and_show_waypoint(full_path, start=start, gpos=gpos)
            input("Press Enter to continue to the next file...")
        except Exception as e:
            print(f"Error loading {fname}: {e}")

# Example usage
if __name__ == "__main__":
    folder = "./rgbd_npy_50"  # replace with your folder path
    run_on_folder(folder)
