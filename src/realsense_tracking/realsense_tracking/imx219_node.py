import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('realsense_tracking')
        
        # Create a publisher for the /2dod/img topic
        self.publisher_ = self.create_publisher(Image, '/vision_2d/img', 10)
        
        # Initialize OpenCV video capture
        self.bridge = CvBridge()
        self.video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

        if not self.video_capture.isOpened():
            self.get_logger().error('Error: Unable to open camera')
            return
        
        # Create a timer to publish images
        self.timer = self.create_timer(1.0 / 30, self.publish_frame)  # 30 FPS

    def publish_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_.publish(msg)
            self.get_logger().info('Published image to /2dod/img')
        else:
            self.get_logger().error('Failed to capture image')

    def destroy_node(self):
        self.video_capture.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
