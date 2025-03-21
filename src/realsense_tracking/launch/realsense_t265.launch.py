import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # RealSense T265 camera driver
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='realsense_camera',
            output='screen',
            parameters=[{
                "enable_pose": True,   # Enable tracking for T265
                "enable_gyro": True,
                "enable_accel": True,
                "publish_odom_tf": True,  # Publishes odometry TF
            }]
        ),

        # Your tracking node
        Node(
            package='realsense_tracking',
            executable='tracking_node',
            name='tracking_node',
            output='screen',
        ),
    ])
