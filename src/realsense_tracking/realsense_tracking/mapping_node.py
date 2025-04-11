import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import threading
import queue

H, W = 200, 752
DISTANCE = 10.0
VOXEL_SIZE = 0.3

class RGBDMapper(Node):
    def __init__(self):
        super().__init__('rgbd_mapper')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/rgbd_data',
            self.listener_callback,
            10
        )
        self.lock = threading.Lock()
        self.accumulated_pcd = o3d.geometry.PointCloud()
        self.frame_count = 0

    def listener_callback(self, msg):
        with self.lock:
            flat = np.array(msg.data, dtype=np.float32)
            if flat.size != H * W * 4:
                self.get_logger().warn("Unexpected RGBD frame size")
                return

            rgbd = flat.reshape((H, W, 4))
            color = rgbd[..., :3] / 255.0
            depth = np.clip(rgbd[..., 3], 0, DISTANCE)

            fx = fy = 286.1167907714844
            cx, cy = W / 2, H / 2
            xx, yy = np.meshgrid(np.arange(W), np.arange(H))
            X = (xx - cx) * depth / fx
            Y = (yy - cy) * depth / fy
            Z = depth

            points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
            colors = color.reshape(-1, 3)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcd.transform([[1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, 1]])

            self.accumulated_pcd += pcd
            self.frame_count += 1
            self.get_logger().info(f"Integrated frame {self.frame_count}")


def visualize_static_map(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window("Static Map")
    voxel_map = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, VOXEL_SIZE)
    vis.add_geometry(voxel_map)
    vis.run()
    vis.destroy_window()

def main(args=None):
    rclpy.init(args=args)
    mapper_node = RGBDMapper()
    ros_thread = threading.Thread(target=rclpy.spin, args=(mapper_node,), daemon=True)
    ros_thread.start()

    try:
        input("Press Enter to stop mapping and visualize...\n")
    finally:
        rclpy.shutdown()
        ros_thread.join()
        mapper_node.get_logger().info("Generating static map...")
        visualize_static_map(mapper_node.accumulated_pcd)

if __name__ == '__main__':
    main()
