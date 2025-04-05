import numpy as np
import open3d as o3d
import cv2

DISTANCE = 10.0

def plan_and_show_waypoint(fp, gpos=np.array([1.0, 0.0, 0.0]), depth_threshold=3.0, occupancy_threshold=10):
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
    voxel_size = 0.2
    occupancy = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    vis.add_geometry(occupancy)

    # Plot gpos as a bigger sphere (goal position)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)  # Larger sphere
    sphere.paint_uniform_color([0, 1, 0])  # Green color for the goal position
    sphere.translate(gpos)
    vis.add_geometry(sphere)

    # Add global coordinate axes at the origin (for XYZ axes)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0, 0, 0]))
    vis.add_geometry(axis)

    # Plan waypoint based on gpos
    direction = gpos / np.linalg.norm(gpos)
    waypoint = None
    best_score = -np.inf
    for i in range(1, 50):  # Search in steps
        wp_candidate = direction * i * 0.1  # Check point in the gpos direction

        # Calculate the voxel grid index for the wp_candidate
        voxel_index = np.floor(wp_candidate / voxel_size).astype(int)

        # Get the voxel center position from the voxel index
        voxel_position = voxel_index * voxel_size  # center of voxel
        
        # Compute distances between the wp_candidate and the voxel center
        distance = np.linalg.norm(points - voxel_position, axis=1)

        # Count how many points are within the voxel (using a distance threshold)
        points_in_voxel_count = np.sum(distance < voxel_size)

        # Check if the voxel is unoccupied based on the number of points within it
        if points_in_voxel_count < occupancy_threshold:  # Low points count means unoccupied
            # Look for a high depth value in the vicinity to ensure there's no obstacle
            x, y, z = wp_candidate
            depth_region = depth_image[int(y), int(x)]  # Get depth from depth map

            if depth_region > depth_threshold:  # High depth means no obstacle
                # Compute "score" based on the direction and depth
                score = np.dot(wp_candidate, direction) * depth_region
                if score > best_score:
                    best_score = score
                    waypoint = wp_candidate

    # If a valid waypoint is found, plot it
    if waypoint is not None:
        print(f"Found waypoint at: {waypoint}")
        waypoint_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)  # Larger sphere for gpos
        waypoint_sphere.paint_uniform_color([1, 0, 0])  # Red color
        waypoint_sphere.translate(waypoint)

        # Add waypoint sphere to visualizer
        vis.add_geometry(waypoint_sphere)
        
    else:
        print('Not found!')
    # Finalize visualization
    vis.run()
    vis.destroy_window()

# Run with default gpos
plan_and_show_waypoint("/Users/yuchunfeng/Downloads/rgbd.npy")
