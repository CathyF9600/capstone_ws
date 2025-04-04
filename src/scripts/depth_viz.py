import numpy as np
import open3d as o3d
import cv2

def rgbd():
    # Load depth map
    fp = "rgbd.npy"
    rgbd_data = np.load(fp, allow_pickle=True)
    color_image = rgbd_data[..., :3]  # RGB values
    depth_image = rgbd_data[..., 3]  # Depth values (ensure it's in meters)

    depth_image[depth_image > 1] = 1
    depth_image[depth_image < 0] = 0

    fx = 286.1167907714844  # focal length in x 
    fy = 286.1167907714844  # focal length in y 
    cx = depth_image.shape[1] / 2  # principal point x 752 / 2
    cy = depth_image.shape[0] / 2

    height, width = depth_image.shape
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    X = (xx - cx) * depth_image / fx
    Y = (yy - cy) * depth_image / fy
    Z = depth_image

    points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
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