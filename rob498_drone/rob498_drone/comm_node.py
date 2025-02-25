#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State

class DroneCommNode(Node):
    def __init__(self):
        super().__init__("drone_comm")

        # Create service servers
        self.srv_launch = self.create_service(Trigger, "comm/launch", self.handle_launch)
        self.srv_land = self.create_service(Trigger, "comm/land", self.handle_land)
        self.srv_abort = self.create_service(Trigger, "comm/abort", self.handle_abort)
        self.srv_test = self.create_service(Trigger, "comm/test", self.handle_test)

        # MAVROS clients
        self.arming_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.set_mode_client = self.create_client(SetMode, "/mavros/set_mode")
        
        # VICON Subscriber
        self.pose_sub = self.create_subscription(
            PoseStamped,
            "/vicon/ROB498_Drone/ROB498_Drone",
            self.vicon_callback,
            10
        )
        self.current_pose = PoseStamped()  # Store latest pose
        
        # Publisher for waypoints
        self.pose_publisher = self.create_publisher(PoseStamped, "/mavros/setpoint_position/local", 10)

        self.get_logger().info("Drone communication node started.")

    def vicon_callback(self, msg):
        """Callback function for VICON pose updates"""
        self.current_pose = msg  # Store latest pose
        self.get_logger().info(
            f"VICON Pose Received: x={msg.pose.position.x}, y={msg.pose.position.y}, z={msg.pose.position.z}"
        )

    def publish_waypoint(self, x=0.0, y=0.0, z=2.0):
        """ Publish a target waypoint. Defaults to (0,0,2) to hold position. """
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "map"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        self.pose_publisher.publish(pose)
        
    def handle_launch(self, request, response):
        self.get_logger().info("Launch command received. Taking off...")

        # Set mode to OFFBOARD
        mode_req = SetMode.Request()
        mode_req.custom_mode = "OFFBOARD"
        future = self.set_mode_client.call_async(mode_req)
        self.get_logger().info("OFFBOARD mode request sent.")

        # Arm the drone
        arm_req = CommandBool.Request()
        arm_req.value = True
        future = self.arming_client.call_async(arm_req)
        self.get_logger().info("Arming request sent.")

        return Trigger.Response(success=True, message="Takeoff initiated.")

    def handle_land(self, request, response):
        self.get_logger().info("Land command received. Landing...")

        mode_req = SetMode.Request()
        mode_req.custom_mode = "AUTO.LAND"
        self.set_mode_client.call_async(mode_req)
        self.get_logger().info("Landing mode request sent.")

        return Trigger.Response(success=True, message="Landing initiated.")

    def handle_abort(self, request, response):
        self.get_logger().info("Abort command received! Stopping flight.")

        mode_req = SetMode.Request()
        mode_req.custom_mode = "AUTO.RTL"
        self.set_mode_client.call_async(mode_req)
        self.get_logger().info("Return-to-launch (RTL) mode request sent.")

        return Trigger.Response(success=True, message="Abort initiated.")

    def handle_test(self, request, response):
        self.get_logger().info("Test command received. Hovering...")
        return Trigger.Response(success=True, message="Test acknowledged.")


def main(args=None):
    rclpy.init(args=args)
    node = DroneCommNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
