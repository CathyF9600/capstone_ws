import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from mavros_msgs.srv import SetMode, CommandBool
from mavros_msgs.msg import State

class OffboardNode(Node):
    def __init__(self):
        super().__init__('offboard_node')

        # Publisher for pose
        self.pose_pub = self.create_publisher(PoseStamped, 'mavros/setpoint_position/local', 10)
        self.state_sub = self.create_subscription(State, 'mavros/state', self.state_callback, 10)
        
        # Services
        self.arm_client = self.create_client(CommandBool, 'mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, 'mavros/set_mode')

        # Pose message
        self.pose = PoseStamped()
        self.pose.pose.position.x = 0
        self.pose.pose.position.y = 0
        self.pose.pose.position.z = 2

        # Timer to publish pose
        self.timer = self.create_timer(0.05, self.publish_pose)
        
        self.current_state = State()
        self.is_armed = False
        self.is_offboard = False

    def state_callback(self, msg):
        self.current_state = msg
        self.is_armed = self.current_state.armed
        self.is_offboard = self.current_state.mode == "OFFBOARD"

    def publish_pose(self):
        self.pose_pub.publish(self.pose)

    def set_mode(self, mode):
        if not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('SetMode service not available')
            return False
        request = SetMode.Request()
        request.custom_mode = mode
        future = self.set_mode_client.call_async(request)
        future.add_done_callback(self.set_mode_callback)
        return True

    def set_mode_callback(self, future):
        result = future.result()
        if result.mode_sent:
            self.get_logger().info(f"Set mode to: {result.custom_mode}")
        else:
            self.get_logger().error(f"Failed to set mode")

    def arm(self):
        if not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Arming service not available')
            return False
        request = CommandBool.Request()
        request.value = True
        future = self.arm_client.call_async(request)
        future.add_done_callback(self.arm_callback)
        return True

    def arm_callback(self, future):
        result = future.result()
        if result.success:
            self.get_logger().info("Vehicle armed")
        else:
            self.get_logger().error("Failed to arm vehicle")

 
def main(args=None):
    rclpy.init(args=args)

    offboard_node = OffboardNode()

    # Wait for connection
    while not offboard_node.current_state.connected:
        rclpy.spin_once(offboard_node)

    # Arm the vehicle and set mode to OFFBOARD
    offboard_node.set_mode("OFFBOARD")
    offboard_node.arm()

    # Keep the node running
    rclpy.spin(offboard_node)

    offboard_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
