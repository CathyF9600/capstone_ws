import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
# from launch.substitutions import FindPackageShare
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    container = ComposableNodeContainer(
        name='my_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='depth_image_proc',
                plugin='depth_image_proc::PointCloudXyzNode',
                name='point_cloud_xyz_node',
                remappings=[
                    ('/image_rect', '/stereo/depth'),
                    ('/camera_info', '/stereo/camera_info')
                ]
            ),
        ],
        output='screen',
    )

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

        # Include the container with depth image processing node
        container,

        # Tracking node
        Node(
            package='realsense_tracking',
            executable='tracking_node',
            name='tracking_node',
            output='screen',
        ),
    ])
