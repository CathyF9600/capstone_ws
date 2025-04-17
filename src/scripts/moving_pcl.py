import numpy as np
import open3d as o3d
import cv2
import heapq
from itertools import count
import os
import time


# Constants
DISTANCE = 10.0 
STEP = 0.2
VOXEL_SIZE = 0.1
COLOR_THRESHOLD = 0.4 # color
COLOR_THRESHOLD_PRUNE = 0.8
MAX_DEPTH = 100
PAD_DIST = 0.1
WAYPOINT_RADIUS = 0.20


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
                            if color and color < COLOR_THRESHOLD_PRUNE:
                                return False  # obstacle detected
    return True

# def vplot(path, vis): # vector plot
#     color = [1, 0, 0]
#     if len(path) < 2:
#         print("Path too short to draw.")
#         return
#     # Convert to Open3D-compatible format
#     points = [list(p) for p in path]
#     lines = [[i, i + 1] for i in range(len(points) - 1)]
#     # Create line set object
#     line_set = o3d.geometry.LineSet(
#         points=o3d.utility.Vector3dVector(points),
#         lines=o3d.utility.Vector2iVector(lines)
#     )
#     # Set color of each line
#     colors = [color for _ in lines]
#     line_set.colors = o3d.utility.Vector3dVector(colors)
#     # Add to visualizer
#     vis.add_geometry(line_set)

def vplot(path, vis):  # vector plot with arrows
    color = [1, 0, 0]
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
    colors = [color for _ in lines]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)

    # Add arrow from point[0] to point[1]
    p0 = np.array(points[0])
    p1 = np.array(points[1])
    direction = p1 - p0
    length = np.linalg.norm(direction)

    if length > 0:
        # Normalize direction
        direction_norm = direction / length

        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.03,     # thicker shaft
            cone_radius=0.1,         # bigger head
            cylinder_height=0.7 * length,  # slightly shorter shaft
            cone_height=0.3 * length       # taller arrowhead
        )
        arrow.paint_uniform_color(color)

        # Align z-axis to `direction`
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, direction_norm)
        c = np.dot(z_axis, direction_norm)
        if np.linalg.norm(v) < 1e-6:
            R = np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p0  # Translate to p0

        arrow.transform(T)
        vis.add_geometry(arrow)


def pplot(vis, gpos, color='R'): # point plot
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)  # Larger sphere
    sphere.paint_uniform_color([1, 0, 0])  # Green color for the goal position
    sphere.translate(gpos)
    vis.add_geometry(sphere)


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


def a_star(pcd, start=np.array([0.0, 0.0, 0.0]),gpos=np.array([0.0, 0.0, -5.0])):
    # Occupancy map (Voxel grid)
    occupancy = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, VOXEL_SIZE)
    voxels = occupancy.get_voxels() # computationally expensive but i have no choice
    voxel_map = build_voxel_index_map(voxels=voxels)
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
                color = get_voxel_color_fast(voxel_map, v_idx)
                if color:
                    print(f'obstacle found at {color:.2f}')
                    if color > COLOR_THRESHOLD: # voxel intensity threhold
                        print(f'obstacle found at {color:.2f}')
                        continue
            else:
                print('v_idx is None!!')
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
        print("Path:", waypoint)
        return waypoint
    else:
        print('Not found!')
        return None


def a_star(pcd, start=np.array([0.0, 0.0, 0.0]),gpos=np.array([2, 0, -5]), prune=False):
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, VOXEL_SIZE)
        voxel_map = build_voxel_index_map(voxel_grid.get_voxels())

        # Set start and goal (e.g., goal 2m ahead of drone)
        # start = np.array([pose[4], pose[5], pose[6]])
        # goal = start + np.array([-1.2, 0.0, -2.8])

        # Short-Horizon Anytime A* path planning
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
                # print("Max depth reached, stopping the pathfinding.")
                break

            if np.linalg.norm(current_pos - goal) < 0.1:
                # print("Path found!")
                found = True
                # print('self.path_points', self.path_points)
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
            target_pos = current_pos #np.array([self.visionipose.pose.position.x, self.vision_pose.pose.position.y, self.vision_pose.pose.position.z])
            print("Reached goal!")
        else:
            target_pos = best_pos
            print(f"Depth limit reached — using best-effort path (dist {best_dist:.2f})")
        waypoint = [target_pos]
        while tuple(waypoint[-1]) in came_from:
            waypoint.append(came_from[tuple(waypoint[-1])])

        # Waypoint visualization
        waypoint.reverse()
        # print("Original Path:", len(waypoint))

        # Path pruning
        if prune:
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
            waypoint = pruned_path
        
        return waypoint
        new_points = add_progress_point(waypoint, self.global_path, full_goal=self.goal)
        if new_points is not None and new_points.all():
            self.path_points.append(new_points)
            return new_points
        return None
    
def set_top_down_view(vis, zoom_level=0.1, distance=3.0):

    # # Get the view control (make sure the visualizer is properly initialized)
    ctr = vis.get_view_control()
    ctr.set_zoom(0.3)
    # parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2025-04-11-23-27-55.json")
    # ctr.convert_from_pinhole_camera_parameters(parameters)


# --- Visualization Class ---
class LiveVisualizer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Floating Block within 1m", width=800, height=600)

        self.pcd = o3d.geometry.PointCloud()
        self.is_added = False

    def load_camera_view(self, path):
        ctr = self.vis.get_view_control()
        parameters = o3d.io.read_pinhole_camera_parameters(path)
        ctr.convert_from_pinhole_camera_parameters(parameters)
        ctr.set_zoom(0.1)  # Optional zoom override

    def update(self, points, colors):
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        waypoint = a_star(self.pcd)
        # Open3D convention
        self.pcd.transform([[1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]])

        if not self.is_added:
            self.vis.add_geometry(self.pcd)
            self.is_added = True
        else:
            self.vis.update_geometry(self.pcd)
            
        # waypoint = a_star(self.pcd)
        if waypoint:
            vplot(waypoint[:2][::-1], self.vis)

        set_top_down_view(self.vis)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()


def run_on_folder(folder_path, start=np.array([0.0, 0.0, 0.0]), gpos=np.array([-2.0, 0.0, -5.0])):
    global counter
    npy_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])
    if not npy_files:
        print("No .npy files found in the folder.")
        return

    vis = LiveVisualizer()

    # vis.add_geometry(sphere)

    while True:
        for fname in npy_files:
                full_path = os.path.join(folder_path, fname)
                print(f"\nShowing: {full_path}")
                rgbd_data = np.load(full_path, allow_pickle=True)

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

                vis.update(points, colors)
                ctr = vis.vis.get_view_control()
                ctr.set_zoom(0.1)
                # parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2025-04-11-23-27-55.json")
                # ctr.convert_from_pinhole_camera_parameters(parameters)
                # if counter == 0:
                #     print('Press Enter to start moving...')
                #     while True:
                #         vis.vis.poll_events()
                #         vis.vis.update_renderer()
                #         if cv2.waitKey(10) == 13:  # Enter key
                #             break
                # else:
                # time.sleep(1)
            # counter += 1


    vis.close()
            # vis.run()
            # vis.destroy_window()

# Example usage
if __name__ == "__main__":
    folder = "./rgbd_npy_50"  # Replace with your folder
    run_on_folder(folder)
