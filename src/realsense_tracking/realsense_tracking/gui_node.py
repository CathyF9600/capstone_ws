import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
import threading
import time

DISTANCE = 3.0
FX = FY = 286.1167907714844  # Intrinsics
CX = 320  # width / 2
CY = 240  # height / 2

class DepthToPointCloud(Node):
    def __init__(self):
        super().__init__('depth_to_pcl')
        self.bridge = CvBridge()

        self.depth_sub = self.create_subscription(Image, '/depth_image', self.depth_callback, 10)
        self.rgb_sub = self.create_subscription(Image, '/rgb_image', self.rgb_callback, 10)

        self.depth_img = None
        self.rgb_img = None

        self.lock = threading.Lock()
        self.vis_thread = threading.Thread(target=self.visualize_loop)
        self.vis_thread.daemon = True
        self.vis_thread.start()

    def depth_callback(self, msg):
        with self.lock:
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')  # float32, meters

    def rgb_callback(self, msg):
        with self.lock:
            self.rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

    def visualize_loop(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window("Live Depth PCL", width=800, height=600)
        pcd = o3d.geometry.PointCloud()
        added = False

        frame = 0

        while True:
            time.sleep(0.03)
            with self.lock:
                if self.depth_img is None or self.rgb_img is None:
                    continue
                depth = np.clip(self.depth_img.copy(), 0, DISTANCE)
                color = self.rgb_img.copy()

            h, w = depth.shape
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            X = (xx - CX) * depth / FX
            Y = (yy - CY) * depth / FY
            Z = depth
            points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
            colors = color.reshape(-1, 3) / 255.0

            # Flip and assign directly
            points = (np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ points.T).T
            
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            if not added:
                vis.add_geometry(pcd)
                added = True
            else:
                vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()
            frame += 1


def main(args=None):
    rclpy.init(args=args)
    node = DepthToPointCloud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
