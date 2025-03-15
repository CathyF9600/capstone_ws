    def publish_target_waypoint(self, target_wp):
        """Publishes a target waypoint to the MAVROS setpoint position topic."""
        waypoint_msg = PoseStamped()
        waypoint_msg.header.stamp = self.get_clock().now().to_msg()
        waypoint_msg.header.frame_id = "map"
        waypoint_msg.pose = target_wp

        self.pose_publisher.publish(waypoint_msg)
        self.get_logger().info(f"Published waypoint: x={target_wp.position.x}, y={target_wp.position.y}, z={target_wp.position.z}")
