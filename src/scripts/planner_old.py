import numpy as np
import heapq
import open3d as o3d

DISTANCE = 10.0
VOXEL_SIZE = 0.2  # You can adjust this depending on your voxel resolution
OCCUPANCY_THRESHOLD = 1  # Threshold for deciding if a voxel is occupied (based on points in the voxel)

def heuristic(a, b):
    # Manhattan distance heuristic (you can use other heuristics depending on the use case)
    return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1]) + np.abs(a[2] - b[2])

def plan_and_show_shortest_path(fp, start_pos=np.array([0.0, 0.0, 0.0]), goal_pos=np.array([1.0, 1.0, 0.0])):
    # Load the RGBD data
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
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="RGB-D Point Cloud", width=800, height=600)

    # Add point cloud to visualizer
    vis.add_geometry(pcd)

    # Occupancy map (Voxel grid)
    occupancy = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, VOXEL_SIZE)

    # A* Algorithm
    open_list = []  # Priority queue (min-heap)
    heapq.heappush(open_list, (heuristic(start_pos, goal_pos), 0, start_pos))  # (f, g, position)

    g_score = {tuple(start_pos): 0}  # g_score of start position
    f_score = {tuple(start_pos): heuristic(start_pos, goal_pos)}  # f_score of start position
    came_from = {}  # To store the path

    # # Directions for neighbor exploration (6 directions: up, down, left, right, forward, backward)
    # directions = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])

    # Directions for neighbor exploration (4 directions: left, right, forward, backward)
    directions = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    # Explore neighbors
    while open_list:
        # Get the position with the lowest f_score
        _, current_g, current_pos = heapq.heappop(open_list)

        if np.allclose(current_pos, goal_pos, atol=VOXEL_SIZE):
            # If we reached the goal, reconstruct the path
            path = []
            while tuple(current_pos) in came_from:
                path.append(current_pos)
                current_pos = came_from[tuple(current_pos)]
            path.append(start_pos)
            path = path[::-1]  # Reverse path to start -> goal
            print("Path found:", path)
            break

        for direction in directions:
            neighbor = current_pos + direction * VOXEL_SIZE  # Move to neighbor
            neighbor_tuple = tuple(neighbor)

            # Check if the neighbor is unoccupied (based on occupancy map)
            voxel_index = np.floor(neighbor / VOXEL_SIZE).astype(int)
            voxel_position = voxel_index * VOXEL_SIZE
            distance = np.linalg.norm(points - voxel_position, axis=1)
            points_in_voxel_count = np.sum(distance < VOXEL_SIZE)

            if points_in_voxel_count < OCCUPANCY_THRESHOLD:  # If unoccupied
                # Calculate the tentative g score
                tentative_g_score = current_g + np.linalg.norm(direction) * VOXEL_SIZE

                # If it's better, record it
                if neighbor_tuple not in g_score or tentative_g_score < g_score[neighbor_tuple]:
                    came_from[neighbor_tuple] = current_pos
                    g_score[neighbor_tuple] = tentative_g_score
                    f_score[neighbor_tuple] = tentative_g_score + heuristic(neighbor, goal_pos)
                    print('tentative_g_score', f_score, neighbor_tuple, tentative_g_score, neighbor)
                    # Add neighbor to the open list
                    heapq.heappush(open_list, (f_score[neighbor_tuple], tentative_g_score, neighbor))

    # Finalize visualization
    vis.run()
    vis.destroy_window()

# Run
plan_and_show_shortest_path("/Users/yuchunfeng/Downloads/rgbd.npy")
