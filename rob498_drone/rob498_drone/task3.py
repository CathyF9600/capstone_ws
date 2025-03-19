#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, Quaternion, PoseArray
from nav_msgs.msg import Odometry
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy, qos_profile_system_default
import numpy as np
import time

WAYPOINT_RADIUS = 0.20  # 40 cm tolerance on handout, 20 cm for better accuracy
TEST_ALTITUDE = 1.5  # Hover altitude
MAX_TEST_TIME = 90  # Maximum test duration
WAYPOINT_TIMEOUT = 10  # Maximum time to reach a waypoint

def quaternion_multiply(q1, q2):
    """Perform quaternion multiplication."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

class DroneCommNodeTask3(Node):
    def __init__(self):
        super().__init__("drone_comm")
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )
        # Store latest pose (default to None)
        self.latest_pose = None
        self.source = None  # 'vicon' or 'realsense'
        self.state = State()
        self.waypoints = None
        self.current_waypoint_idx = 0
        self.start_time = None  # To track test time

        # Create service servers
        self.srv_launch = self.create_service(Trigger, "rob498_drone_3/comm/launch", self.handle_launch)
        self.srv_land = self.create_service(Trigger, "rob498_drone_3/comm/land", self.handle_land)
        self.srv_abort = self.create_service(Trigger, "rob498_drone_3/comm/abort", self.handle_abort)
        self.srv_test = self.create_service(Trigger, "rob498_drone_3/comm/test", self.handle_test)

        self.create_subscription(State, '/mavros/state', self.state_cb, qos_profile_system_default)

        # VICON Subscriber
        self.vicon_sub = self.create_subscription(
            PoseStamped,
            "/vicon/ROB498_Drone/ROB498_Drone",
            self.vicon_callback,
            qos_profile_system_default
        )

        # RealSense Subscriber
        self.realsense_sub = self.create_subscription(
            Odometry,
            "/camera/pose/sample",
            self.realsense_callback,
            qos_profile_system_default
        )

        # ego publisher
        self.ego_pub = self.create_publisher(PoseStamped,'/mavros/vision_pose/pose', qos_profile_system_default) # "/vicon/ROB498_Drone/ROB498_Drone", 10)
        # Timer to publish waypoints at 20 Hz
        self.create_timer(1/20, self.publish_vision_pose)

        # Publisher for MAVROS setpoints
        self.setpoint_publisher = self.create_publisher(PoseStamped, "/mavros/setpoint_position/local", qos_profile_system_default)
        # MAVROS clients
        self.arming_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.set_mode_client = self.create_client(SetMode, "/mavros/set_mode")
        self.timer = self.create_timer(0.02, self.cmdloop_callback) # set arm and offboard mode if haven't
        
        if self.latest_pose:
            self.hover_pose = self.latest_pose
        else:
            self.hover_pose = PoseStamped()
        self.update_hover_pose(0.0, 0.0, 0.0)

        self.get_logger().info("Drone communication node started. Listening to VICON and RealSense.")

        self.get_logger().info("Waiting for waypoints...")
        self.subscription = self.create_subscription(
            PoseArray,
            "rob498_drone_3/comm/waypoints",
            self.waypoints_callback,
            10
        )
        self.test_start = False
        self.wp_received = False
        self.create_timer(1/20, self.publish_target_waypoint)


    def waypoints_callback(self, msg): # special for task3
        """Receives the list of waypoints."""
        if not self.wp_received:
            self.waypoints = msg.poses
            self.current_waypoint_idx = 0
            self.get_logger().info(f"Received {len(self.waypoints)} waypoints.")
            self.wp_received = True


    def state_cb(self, msg):
        self.state = msg
        self.get_logger().info(f"Current mode: {self.state.mode}")
    
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
            
    def cmdloop_callback(self):
        if self.state.mode != "OFFBOARD":
            self.set_mode("OFFBOARD")
        if not self.state.armed:
            self.arm(True)
        
    def vicon_callback(self, msg):
        """Update pose from VICON."""
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        
        self.latest_pose = msg
        self.source = "vicon"
        # self.vicon_pub.publish(msg)
        # self.get_logger().info(f"VICON Pose Received: x={msg.pose.position.x}, y={msg.pose.position.y}, z={msg.pose.position.z}")

    def realsense_callback(self, msg):
        """Update pose from RealSense with 90-degree yaw rotation."""
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        current_pose_d = PoseStamped()
        current_pose_d.header.stamp = self.get_clock().now().to_msg()
        current_pose_d.header.frame_id = "map"
        current_pose_d.pose = msg.pose.pose

        # Original orientation
        q_orig = np.array([
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z
        ])
        self.get_logger().info(f"Cam Original: x={msg.pose.pose.orientation.x:.3f}, y={msg.pose.pose.orientation.y:.3f}, z={msg.pose.pose.orientation.z:.3f}")
        # 90-degree yaw quaternion (0, 0, sin(π/4), cos(π/4))
        # q_yaw = np.array([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])

        # Apply rotation
        q_new = q_orig #quaternion_multiply(q_yaw, q_orig)

        # Update orientation
        current_pose_d.pose.orientation = Quaternion(
            w=float(q_new[0]),
            x=float(q_new[1]),
            y=float(q_new[2]),
            z=float(q_new[3])
        )
        # self.get_logger().info(f"Cam Converted: x={current_pose_d.pose.orientation.x}, y={current_pose_d.pose.orientation.y}, z={current_pose_d.pose.orientation.z}")

        self.latest_pose = current_pose_d
        self.source = "realsense"

    def update_hover_pose(self, x, y, z):
        self.hover_pose.header.stamp = self.get_clock().now().to_msg()
        self.hover_pose.header.frame_id = "map"
        self.hover_pose.pose.position.x = x
        self.hover_pose.pose.position.y = y
        self.hover_pose.pose.position.z = z
        # self.setpoint_publisher.publish(self.hover_pose)

        
    def publish_vision_pose(self):
        if self.latest_pose:
            # hover_pose_d = PoseStamped()
            cur_ego_pose = self.latest_pose
            cur_ego_pose.header.stamp = self.get_clock().now().to_msg()
            cur_ego_pose.header.frame_id = "map"
        
            self.get_logger().info(f"Latest pose: x={cur_ego_pose.pose.position.x:.3f}, y={cur_ego_pose.pose.position.y:.3f}, z={cur_ego_pose.pose.position.z:.3f}")

            self.ego_pub.publish(cur_ego_pose)
            self.get_logger().info(f"Published itself's pose from {self.source}.")
        else:
            self.get_logger().info(f"Published itself's pose from nowhere!")
            

    def handle_launch(self, request, response):
        self.get_logger().info("Launch command received. Taking off...")

        if self.state.mode != "OFFBOARD":
            self.set_mode("OFFBOARD")
        if not self.state.armed:
            self.arm(True)
        # Change the altitude
        self.hover_pose.header.stamp = self.get_clock().now().to_msg()
        # Ascend to 1.5 meters upon launching
        target_height = 1.5
        self.hover_pose.pose.position.z = target_height
        if self.source == 'realsense':
            self.hover_pose.pose.position.z = target_height - 0.1
        # Arm the drone
        arm_req = CommandBool.Request()
        arm_req.value = True
        future = self.arming_client.call_async(arm_req)
        self.get_logger().info("Launch request sent.")
        # Ensure a response is returned
        response.success = True
        response.message = "Takeoff initiated."
        return response
 
    def handle_land(self, request, response):
        self.get_logger().info("Land command received. Landing...")

        mode_req = SetMode.Request()
        mode_req.custom_mode = "AUTO.LAND"
        future = self.set_mode_client.call_async(mode_req)
        self.hover_pose.header.stamp = self.get_clock().now().to_msg()
        self.hover_pose.pose.position.z = 0.05
        # self.get_logger().info("Landing mode request sent.")
        response.success = True
        return response


    def handle_abort(self, request, response):
        self.get_logger().info("Abort command received! Stopping flight.")

        mode_req = SetMode.Request()
        mode_req.custom_mode = "AUTO.RTL"
        self.set_mode_client.call_async(mode_req)
        self.get_logger().info("Return-to-launch (RTL) mode request sent.")

        return response

    
    def handle_test(self, request, response): # special for task3
        """Handles the TEST command by navigating through waypoints."""
        if not self.waypoints:
            self.get_logger().error("No waypoints received.")
            response.success = False
            response.message = "No waypoints available."
            return response
        self.test_start = True
        return response


    def publish_target_waypoint(self): # special for task3
        """Publishes the given waypoint to MAVROS."""
        if not self.test_start:
            self.setpoint_publisher.publish(self.hover_pose)
            return

        if self.current_waypoint_idx < len(self.waypoints):

            target_wp = self.waypoints[self.current_waypoint_idx]
            self.hover_pose.header.stamp = self.get_clock().now().to_msg()
            self.hover_pose.pose = target_wp
            self.setpoint_publisher.publish(self.hover_pose)

            if self.is_within_waypoint(target_wp):
                self.get_logger().info(f"Reached waypoint {self.current_waypoint_idx + 1}")
                self.current_waypoint_idx += 1
        else:
            self.get_logger().info("All waypoints reached. Hovering at the last waypoint & Ready to land")
            last_waypoint = self.waypoints[-1]
            self.hover_pose.header.stamp = self.get_clock().now().to_msg()
            self.hover_pose.pose = last_waypoint
            self.setpoint_publisher.publish(self.hover_pose)


    def is_within_waypoint(self, waypoint): # special for task3
        """Checks if the drone is within the target waypoint radius."""
        if not self.latest_pose:
            self.get_logger().info(f"self.latest_pose is null")
            return False

        dx = self.latest_pose.pose.position.x - waypoint.position.x
        dy = self.latest_pose.pose.position.y - waypoint.position.y
        dz = self.latest_pose.pose.position.z - waypoint.position.z
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        self.get_logger().info(f"Distance to "
                    f"({waypoint.position.x:.1f}, {waypoint.position.y:.1f}, {waypoint.position.z:.1f}) is "
                    #    f"x={self.latest_pose.pose.position.x:.3f}, y={self.latest_pose.pose.position.y:.3f}, z={self.latest_pose.pose.position.z:.3f}
                    f"{distance:.1f}")
        return distance <= WAYPOINT_RADIUS


import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose

class WaypointPublisher(Node):
    def __init__(self):
        super().__init__('waypoint_publisher')
        
        # Create publisher
        self.publisher_ = self.create_publisher(PoseArray, 'rob498_drone_3/comm/waypoints', 10)
        
        # Timer to publish waypoints
        self.timer = self.create_timer(1.0, self.publish_waypoints)  # Publishes every 1 second
        
        self.get_logger().info("Waypoint Publisher Node Started")

    def publish_waypoints(self):
        # Define waypoints (x, y, z)
        waypoints = [
            (0.0, 0.0, 2.0),
            (2.0, 2.0, 2.0),
            (4.0, 0.0, 2.0),
            (2.0, -2.0, 2.0),
            (0.0, 0.0, 2.0)
        ]
        
        # Create PoseArray message
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "map"  # Change if needed
        
        for x, y, z in waypoints:
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            pose.orientation.w = 1.0  # Neutral orientation
            pose_array.poses.append(pose)

        # Publish waypoints
        self.publisher_.publish(pose_array)
        self.get_logger().info(f"Published {len(waypoints)} waypoints.")

from rclpy.executors import MultiThreadedExecutor

# def main(args=None):
#     rclpy.init(args=args)

#     publisher_node = WaypointPublisher()
#     subscriber_node = DroneCommNodeTask3()

#     executor = MultiThreadedExecutor()
#     executor.add_node(publisher_node)
#     executor.add_node(subscriber_node)

#     try:
#         executor.spin()
#     finally:
#         publisher_node.destroy_node()
#         subscriber_node.destroy_node()
#         rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = DroneCommNodeTask3()
    rclpy.spin(node)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
