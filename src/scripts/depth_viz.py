import numpy as np
import open3d as o3d
import cv2
from skimage.draw import disk

DISTANCE = 5
THRESHOLD = 10

class Grid():
    def __init__(self):
        self.map_settings_dict = {"origin": (0,0),
                                   "resolution": 0.25}
        self.map_shape=(5, 10)
        self.robot_radius = 0.5
        
        
    def point_to_cell(self, point):
        #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        #point is a 2 by N matrix of points of interest
        # print("TO DO: Implement a method to get the map cell the robot is currently occupying")
        origin = np.array(self.map_settings_dict["origin"][:2]).reshape(1,2)
        scale = self.map_settings_dict["resolution"]

        indices = (point - origin) / scale
        indices[1, :] = self.map_shape[0] - indices[1, :] # world frame to grid frame
        # print('np.floor(indices).astype(int)', origin, np.floor(indices).astype(int))
        return np.floor(indices).astype(int)


    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        # print("TO DO: Implement a method to get the pixel locations of the robot path")
        robot_indices_grid = self.point_to_cell(points)
        robot_r_grid = self.robot_radius / self.map_settings_dict["resolution"]
        rr, cc = [], []

        for i in range(0, len(robot_indices_grid[0])):
            center = (robot_indices_grid[0][i], robot_indices_grid[1][i])
            # disk returns the row (rr_o) and column (cc_o) indices of all pixels inside a circular region centered at center with radius robot_r
            rr_o, cc_o = disk(center, robot_r_grid)
            # print('rr', rr_o, cc_o)
            rr_o = np.clip(rr_o, 0, self.map_shape[1] - 1) #limits values to be between 0 and map size -1
            cc_o = np.clip(cc_o, 0, self.map_shape[0] - 1)
            rr = np.concatenate((rr, rr_o)).astype(int)
            cc = np.concatenate((cc, cc_o)).astype(int)

        return rr, cc
    

def build_map(depth_image, points, voxel_size=0.05):
    grid = Grid()
    # Filter Z
    depth_image[depth_image > 5] = 5
    filtered = points

    # Discretize
    xy_voxels = np.floor(filtered[:, :2] / voxel_size).astype(int)

    # Shift to positive indices
    min_xy = xy_voxels.min(axis=0)
    xy_voxels -= min_xy

    # # Count point hits per voxel
    # unique_voxels, counts = np.unique(xy_voxels, axis=0, return_counts=True)
    # counts = counts[counts >= THRESHOLD]
    # # Grid size
    # w, h = unique_voxels.max(axis=0) + 1
    # bev = np.zeros((h, w), dtype=np.float32)

    # # Normalize to [0, 1]
    # probs = counts / 10 #counts.max()

    # for (x, y), p in zip(unique_voxels, probs):
    #     bev[y, x] = p

    rr, cc = grid.points_to_robot_circle(filtered[:, :2])
    print('rr cc', rr, cc)
    h, w = len(rr), len(cc)
    bev = np.zeros((h, w), dtype=np.float32)
    for (x, y) in zip(cc, rr):
        bev[y, x] = 1
    # Scale to [0, 255] and convert to uint8
    bev_img = (bev * 255).astype(np.uint8)

    # # Optional: apply blur for softening
    # bev_img = cv2.GaussianBlur(bev_img, (3, 3), sigmaX=1)

    # Convert to 3-channel grayscale for display
    bev_rgb = cv2.cvtColor(bev_img, cv2.COLOR_GRAY2BGR)

    # Resize for visualization
    bev_resized = cv2.resize(bev_rgb, (w * 5, h * 5), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Grayscale BEV Map", bev_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





def rgbd():
    # Load RGB-D image
    fp = "/Users/yuchunfeng/Downloads/rgbd.npy"
    rgbd_data = np.load(fp, allow_pickle=True)
    color_image = rgbd_data[..., :3]  # RGB
    depth_image = rgbd_data[..., 3]   # Depth in meters

    # Clamp depth for visualization
    depth_image = np.clip(depth_image, 0, DISTANCE)

    # depth_image[depth_image < 0] = 0
    
    # Camera intrinsics
    fx = fy = 286.1167907714844
    cx = depth_image.shape[1] / 2
    cy = depth_image.shape[0] / 2

    # Generate point cloud
    height, width = depth_image.shape
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    X = (xx - cx) * depth_image / fx
    Y = (yy - cy) * depth_image / fy
    Z = depth_image

    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    
    colors = color_image.reshape(-1, 3) / 255.0

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Flip the point cloud to Open3D convention (camera facing +Z)
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

    # build_map(depth_image, np.asarray(pcd.points))

    # Show point cloud in interactive viewer
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="RGB-D Point Cloud",
        width=800,
        height=600,
        point_show_normal=False
    )

    # Also show 2D depth map for reference
    depth_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow("Depth Map", depth_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def vis_pcl():
    fp = "gpoints.npy"
    points = np.load(fp, allow_pickle=True)
    # print(points.T[:, -1])
    points = points.T[:,:3]
    fp = "rgbd.npy"
    rgbd_data = np.load(fp, allow_pickle=True)
    color_image = rgbd_data[..., :3]
    depth_image = rgbd_data[..., 3]  # Depth values (ensure it's in meters)

    depth_image[depth_image > 1] = 1
    depth_image[depth_image < 0] = 0

    # fx = 286.1167907714844  # focal length in x 
    # fy = 286.1167907714844  # focal length in y 
    # cx = depth_image.shape[1] / 2  # principal point x 752 / 2
    # cy = depth_image.shape[0] / 2

    # height, width = depth_image.shape
    # xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    # X = (xx - cx) * depth_image / fx
    # Y = (yy - cy) * depth_image / fy
    # Z = depth_image

    # points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    print(points.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(color_image.reshape(-1, 3) / 255.0)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])

    depth_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Display the depth map image using OpenCV
    cv2.imshow("Depth Map", depth_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# vis_pcl()
rgbd()