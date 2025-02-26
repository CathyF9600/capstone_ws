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
        self.vicon_sub = self.create_subscription(
            PoseStamped,
            "/vicon/ROB498_Drone/ROB498_Drone",
            self.vicon_callback,
            10
        )

        # RealSense Subscriber
        self.realsense_sub = self.create_subscription(
            PoseStamped,
            "/camera/pose/sample",
            self.realsense_callback,
            10
        )

        # Publisher for MAVROS setpoints
        self.pose_publisher = self.create_publisher(PoseStamped, "/mavros/setpoint_position/local", 10)

        # Timer to publish waypoints at 20 Hz
        self.create_timer(1/20, self.publish_waypoint)
        
        # Store latest pose (default to None)
        self.initial_pose = None
        self.latest_pose = None
        self.source = None  # 'vicon' or 'realsense'

        self.get_logger().info("Drone communication node started. Listening to VICON and RealSense.")

    
    def vicon_callback(self, msg):
        """Update pose from VICON."""
        if not self.initial_pose:
            self.initial_pose = msg
        self.latest_pose = msg
        self.source = "vicon"
        self.get_logger().info(f"VICON Pose Received: x={msg.pose.position.x}, y={msg.pose.position.y}, z={msg.pose.position.z}")

    
    def realsense_callback(self, msg):
        """Update pose from RealSense."""
        if not self.initial_pose:
            self.initial_pose = msg
        self.latest_pose = msg
        self.source = "realsense"
        self.get_logger().info(f"RealSense Pose Received: x={msg.pose.position.x}, y={msg.pose.position.y}, z={msg.pose.position.z}")

    '''
    def publish_waypoint(self):
        """Continuously publish the latest pose to MAVROS at 20 Hz."""
        if self.latest_pose is not None:
            self.pose_publisher.publish(self.latest_pose)
            self.get_logger().info(f"Published waypoint from {self.source}")
    '''
    
    def publish_waypoint(self):
    """Continuously publish the latest pose to MAVROS at 20 Hz, forcing a hover height."""
        if self.latest_pose is not None:
            hover_pose = PoseStamped()
            hover_pose.header.stamp = self.get_clock().now().to_msg()
            hover_pose.header.frame_id = "map"
        
            hover_pose.pose.position.x = self.initial_pose.pose.position.x
            hover_pose.pose.position.y = self.initial_pose.pose.position.y
            hover_pose.pose.position.z = 1.5  # Force drone to hover at 2 meters
        
            self.pose_publisher.publish(hover_pose)
            self.get_logger().info(f"Published hover waypoint at z=2.0 from {self.source}")
    
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
