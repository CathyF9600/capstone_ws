import rclpy
from rclpy.node import Node
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped

class PX4Sim(Node):
    def __init__(self):
        super().__init__('task2_sim')
        # Subscribe to PX4 state and pose topics
        self.state_subscriber = self.create_subscription(
            State,
            '/mavros/state',
            self.state_callback,
            10
        )

        self.pose_subscriber = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.pose_callback,
            10
        )
        self.get_logger().info("PX4 Simulation Node Started")

    def state_callback(self, msg):
        self.get_logger().info(f"PX4 State: {msg.mode}, Armed: {msg.armed}")

    def pose_callback(self, msg):
        self.get_logger().info(f"Current Position: {msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z}")

def main(args=None):
    rclpy.init(args=args)
    node = PX4Sim()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
