#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from rclpy.qos import qos_profile_system_default

class DroneCommNode(Node):
    def __init__(self):
        # Book Keeping ######################################################################################################################
        # node name = "drone_comm"
        super().__init__("drone_comm")
        # Policies
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT, # Most reliable
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL, # Transient Local
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST, # Keep Last
            depth=1 # Only the last message is kept
        )

        # Initialize ########################################################################################################################
        # Timers: 
        self.timer = self.create_timer(0.02, self.cmdloop_callback)
        self.create_timer(1/20, self.publish_vision_pose)       # Timer to publish vision poses at 20 Hz
        self.create_timer(1/20, self.publish_waypoint)          # Timer to publish waypoints at 20 Hz

        self.source = None  # 'vicon' or 'realsense'
        self.state = State()
        
        # Pose info - start with NONE and reset to (0,0,0)
        self.initial_pose = None
        self.latest_pose = None
        if self.latest_pose:
            self.hover_pose = self.latest_pose
        else:
            self.hover_pose = PoseStamped()
        self.update_hover_pose(0.0, 0.0, 0.0)
        
        # SERVERS ############################################################################################################################
        # Sends out a task, want response upon completed
        self.srv_launch = self.create_service(Trigger, "rob498_drone_3/comm/launch", self.handle_launch)
        self.srv_land = self.create_service(Trigger, "rob498_drone_3/comm/land", self.handle_land)
        self.srv_abort = self.create_service(Trigger, "rob498_drone_3/comm/abort", self.handle_abort)
        self.srv_test = self.create_service(Trigger, "rob498_drone_3/comm/test", self.handle_test)

        # SUBSCRIBERS ########################################################################################################################
        # Information coming in from yappers I mean Publishers

        # CommNode is subscribed to mavros... self.mavros_sub is a subscriber to mavros
        # Type = State means the subscriber is expecting to receive messages of the State type.
        self.mavros_sub = self.create_subscription(
            State, 
            '/mavros/state', 
            self.state_cb, 
            qos_profile_system_default
        )

        # CommNode is subscribed to VICON... self.vicon_sub is a subscriber to VICON
        # Type = PoseStamped has position and orientation
        self.vicon_sub = self.create_subscription(
            PoseStamped,  # Message type
            "/vicon/ROB498_Drone/ROB498_Drone",  # Topic name
            self.vicon_callback,  # Callback function
            qos_profile_system_default  # QoS (Quality of Service) settings
        )

        # CommNode is subscribed to RealSense... self.realsense_sub is a subscriber to VICON
        # Type = Odometry has linear and angular velocity whereas type = PoseStamped doesn't have those. 
        self.realsense_sub = self.create_subscription(
            Odometry,
            "/camera/pose/sample",
            self.realsense_callback,
            qos_profile_system_default
        )

        # PUBLISHERS ######################################################################################################################
        # Yapping I mean infomation going out
        # dummy publisher
        self.ego_pub = self.create_publisher(
            PoseStamped,
            '/mavros/vision_pose/pose', 
            # "/vicon/ROB498_Drone/ROB498_Drone", 10)
            qos_profile_system_default
        ) 
        
        # Publisher for MAVROS setpoints
        self.pose_publisher = self.create_publisher(
            PoseStamped, 
            "/mavros/setpoint_position/local", 
            qos_profile_system_default
        )

        # CLIENTS ##########################################################################################################################
        # Sending some other node responses when we are done
        # MAVROS clients
        self.arming_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.set_mode_client = self.create_client(SetMode, "/mavros/set_mode")

        # init finished
        self.get_logger().info("Drone communication node started. Listening to VICON and RealSense.")


    # Functionality functions ##############################################################################################################
    def set_mode(self, mode):
        if self.set_mode_client.service_is_ready():
            req = SetMode.Request()
            req.custom_mode = mode
            self.set_mode_client.call_async(req)
    
    def arm(self, arm_status):
        if self.arming_client.service_is_ready():
            req = CommandBool.Request()
            req.value = arm_status
            self.arming_client.call_async(req)

    def update_hover_pose(self, x, y, z):
        self.hover_pose.header.stamp = self.get_clock().now().to_msg()
        self.hover_pose.header.frame_id = "map"
        self.hover_pose.pose.position.x = x
        self.hover_pose.pose.position.y = y
        self.hover_pose.pose.position.z = z


    # Call back function - execute when a message is received #################################################################################
    def state_cb(self, msg):
        self.state = msg
        self.get_logger().info(f"Current mode: {self.state.mode}")
            
    def cmdloop_callback(self):
        if self.state.mode != "OFFBOARD":
            self.set_mode("OFFBOARD")
        if not self.state.armed:
            self.arm(True)
        
    def vicon_callback(self, msg):
        """Update pose from VICON."""
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        if not self.initial_pose:
            self.initial_pose = msg
        self.latest_pose = msg
        self.source = "vicon"
        # self.vicon_pub.publish(msg)
        # self.get_logger().info(f"VICON Pose Received: x={msg.pose.position.x}, y={msg.pose.position.y}, z={msg.pose.position.z}")

    def realsense_callback(self, msg):
        """Update pose from RealSense."""
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        current_pose_d = PoseStamped()
        current_pose_d.header.stamp = self.get_clock().now().to_msg()
        current_pose_d.header.frame_id = "map"
    
        current_pose_d.pose = msg.pose.pose

        if not self.initial_pose:
            self.initial_pose = current_pose_d
        self.latest_pose = current_pose_d
        self.source = "realsense"

        # .position.x = 4.2 #self.initial_pose.pose.position.x
        # current_pose_d.pose.position.y = 4.2 # self.initial_pose.pose.position.y
        # current_pose_d.pose.position.z = 0.0  # Force drone to hover at 2 meters
        # current_pose_d.pose = self.latest_pose

        # self.vicon_pub.publish(current_pose_d)
        # self.get_logger().info(f"RealSense Pose Received: x={msg.pose.pose.x}, y={msg.pose.pose.y}, z={msg.pose.pose.z}")
    



    # Publisher functions ###########################################################################################################
    # Fancy FYIs
    def publish_vision_pose(self):
        if self.latest_pose:
            # hover_pose_d = PoseStamped()
            hover_pose_d = self.latest_pose
            hover_pose_d.header.stamp = self.get_clock().now().to_msg()
            hover_pose_d.header.frame_id = "map"
        
            # hover_pose_d.pose.position.x = 4.2 #self.initial_pose.pose.position.x
            # hover_pose_d.pose.position.y = 4.2 # self.initial_pose.pose.position.y
            # hover_pose_d.pose.position.z = 0.0  # Force drone to hover at 2 meters
            # hover_pose_d.pose = self.latest_pose
            self.get_logger().info(f"Latest pose: x={hover_pose_d.pose.position.x}, y={hover_pose_d.pose.position.y}, z={hover_pose_d.pose.position.z}")

            self.ego_pub.publish(hover_pose_d)
            self.get_logger().info(f"Published itself's pose from {self.source}.")
        else:
            self.get_logger().info(f"Published itself's pose from nowhere!")
            
    def publish_waypoint(self):
        """Continuously publish the latest pose to MAVROS at 20 Hz, forcing a hover height."""
        # mode_req = SetMode.Request()
        self.hover_pose.header.stamp = self.get_clock().now().to_msg()
        self.pose_publisher.publish(self.hover_pose)
        self.get_logger().info(f"Test cmd Received: x={self.hover_pose.pose.position.x}, y={self.hover_pose.pose.position.y}, z={self.hover_pose.pose.position.z}")
        # self.get_logger().info(f"Published hover waypoint at (0,0,0) from {self.source}, in {mode_req.custom_mode}")
    
    # TRIGGER POINTS ###############################################################################################################
    # Make the drone do shit
    def handle_launch(self, request, response):
        self.get_logger().info("Launch command received. Taking off...")
        
        # Change the altitude
        self.hover_pose.header.stamp = self.get_clock().now().to_msg()
        self.hover_pose.pose.position.z = 1.5  # Force drone to hover at 2 meters   
        if self.source == 'realsense':
            self.hover_pose.pose.position.z = 1.5 - 0.15

        # Take off complete affirmation
        self.get_logger().info("Launch request sent.")
        # return Trigger.Response(success=True, message="Takeoff initiated.")
        # Ensure a response is returned
        response.success = True
        response.message = "Takeoff initiated."
        return response  # <-- Add this
 
    def handle_land(self, request, response):
        self.get_logger().info("Land command received. Trying to land...")

        mode_req = SetMode.Request()
        mode_req.custom_mode = "AUTO.LAND"
        future = self.set_mode_client.call_async(mode_req)
        # while not self.set_mode_client.wait_for_service(timeout_sec=5.0):
        #     self.get_logger().warn('Waiting for set_mode service...')
        self.hover_pose.header.stamp = self.get_clock().now().to_msg()
        self.hover_pose.pose.position.z = 0.2 
        self.get_logger().info("Landing mode request sent.")
        response.success = True
        response.message = "Landing initiated"
        return response # Trigger.Response(success=True, message="Landing initiated.")


    def handle_abort(self, request, response):
        self.get_logger().info("Abort command received! Stopping flight.")

        mode_req = SetMode.Request()
        mode_req.custom_mode = "AUTO.RTL"
        self.set_mode_client.call_async(mode_req)
        self.get_logger().info("Return-to-launch (RTL) mode request sent.")

        return Trigger.Response(success=True, message="Aborted")

    
    def handle_test(self, request, response):
        self.hover_pose.header.stamp = self.get_clock().now().to_msg()
        # self.hover_pose.header.frame_id = "map"
        # self.hover_pose.pose.position.x = 0.0 #self.initial_pose.pose.position.x
        # self.hover_pose.pose.position.y = 0.0 # self.initial_pose.pose.position.y
        self.hover_pose.pose.position.z = 1.5  # Force drone to hover at 2 meters   
        if self.source == 'realsense':
            self.hover_pose.pose.position.z = 1.5 - 0.15  # Force drone to hover at 2 meters   
        # for i in range(1000):
        #     self.get_logger().info(f"Test cmd Received: x={self.hover_pose.pose.position.x}, y={self.hover_pose.pose.position.y}, z={self.hover_pose.pose.position.z}")
        return Trigger.Response(success=True, message="Test acknowledged.")


def main(args=None):
    rclpy.init(args=args)
    node = DroneCommNode()
    rclpy.spin(node)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
