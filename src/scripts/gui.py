import numpy as np
import open3d as o3d
import time

# use synthetic rgbd to demonstrate live 3D mapping

DISTANCE = 3.0  # Max clipping distance

# --- Visualization Class ---
class LiveVisualizer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Floating Block within 1m", width=800, height=600)
        self.pcd = o3d.geometry.PointCloud()
        self.is_added = False

    def update(self, points, colors):
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        if not self.is_added:
            self.vis.add_geometry(self.pcd)
            self.is_added = True
        else:
            self.vis.update_geometry(self.pcd)

        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()


# --- Main ---
def simulate_moving_block():
    vis = LiveVisualizer()

    # Load RGB-D data
    rgbd_data = np.load("/Users/yuchunfeng/Downloads/rgbd.npy", allow_pickle=True)
    color_img = rgbd_data[..., :3]
    depth_img = np.clip(rgbd_data[..., 3], 0, DISTANCE)

    h, w = depth_img.shape
    fx = fy = 286.1167907714844
    cx = w / 2
    cy = h / 2

    # Compute base point cloud
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    X = (xx - cx) * depth_img / fx
    Y = (yy - cy) * depth_img / fy
    Z = depth_img
    all_points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    all_colors = color_img.reshape(-1, 3) / 255.0

    # Flip to Open3D convention
    flip = np.array([[1, 0, 0],
                     [0, -1, 0],
                     [0, 0, -1]])
    all_points = (flip @ all_points.T).T

    # Identify the <1m block
    mask = all_points[:, 2] < 0.5 #1.0
    dynamic_points = all_points[mask]
    dynamic_colors = all_colors[mask]
    static_points = all_points[~mask]
    static_colors = all_colors[~mask]

    try:
        for frame in range(100):
            # Translate dynamic block
            dx = 0.1 * np.sin(frame * 0.1) * 5
            dy = 0.05 * np.cos(frame * 0.1) * 5
            dz = 0.02 * np.sin(frame * 0.1) * 5
            moved_block = dynamic_points + np.array([dx, dy, dz])

            # Merge with static points
            combined_points = np.vstack((static_points, moved_block))
            combined_colors = np.vstack((static_colors, dynamic_colors))

            vis.update(combined_points, combined_colors)
            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        vis.close()


if __name__ == "__main__":
    simulate_moving_block()
