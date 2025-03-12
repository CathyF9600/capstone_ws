import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        LogInfo(
            msg="Launching PX4 with Gazebo"
        ),
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-entity', 'my_drone', '-file', '/path/to/your/robot_model.urdf'],
            output='screen'
        ),
        Node(
            package='rob498_sim',
            executable='task2_sim',
            name='task2_sim',
            output='screen',
            parameters=[{'robot_description': '/path/to/robot_model.urdf'}]
        ),
        Node(
            package='px4',
            executable='px4',
            name='px4_node',
            output='screen',
            arguments=['/path/to/your/px4_sitl_default']
        ),
    ])
