import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger, Trigger_Response

class CommNode(Node):
    def __init__(self):
        super().__init__('comm_node')

        # Create services
        self.srv_launch = self.create_service(Trigger, '/comm/launch', self.launch_callback)
        self.srv_test = self.create_service(Trigger, '/comm/test', self.test_callback)
        self.srv_land = self.create_service(Trigger, '/comm/land', self.land_callback)
        self.srv_abort = self.create_service(Trigger, '/comm/abort', self.abort_callback)

        self.get_logger().info("Drone communication node started")

    def launch_callback(self, request, response):
        # Define logic for launch
        response.success = True
        response.message = "Launch command received"
        return response

    def test_callback(self, request, response):
        # Define logic for test
        response.success = True
        response.message = "Test command received"
        return response

    def land_callback(self, request, response):
        # Define logic for land
        response.success = True
        response.message = "Land command received"
        return response

    def abort_callback(self, request, response):
        # Define logic for abort
        response.success = True
        response.message = "Abort command received"
        return response

def main(args=None):
    rclpy.init(args=args)
    node = CommNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
