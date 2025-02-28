# In your launch file (start_offb.launch.py)
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='offboard_py',
            executable='offb_node.py',
            output='screen',
            parameters=[{'param_name': 'param_value'}]
        )
    ])
