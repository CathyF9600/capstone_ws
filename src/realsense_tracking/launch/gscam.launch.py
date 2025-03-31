from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    left_camera = LaunchConfiguration('left_camera', default='0')
    right_camera = LaunchConfiguration('right_camera', default='1')
    width = LaunchConfiguration('width', default='3280')
    height = LaunchConfiguration('height', default='2464')
    fps = LaunchConfiguration('fps', default='20/1')
    format = LaunchConfiguration('format', default='NV12')

    return LaunchDescription([
        DeclareLaunchArgument('left_camera', default_value='0', description='Left camera sensor ID'),
        DeclareLaunchArgument('right_camera', default_value='1', description='Right camera sensor ID'),
        DeclareLaunchArgument('width', default_value='3280', description='Camera image width'),
        DeclareLaunchArgument('height', default_value='2464', description='Camera image height'),
        DeclareLaunchArgument('fps', default_value='20/1', description='Camera framerate'),
        DeclareLaunchArgument('format', default_value='NV12', description='Camera image format'),

        Node(
            package='gscam',
            executable='gscam',
            name='left_camera',
            namespace='left',
            parameters=[{
                'camera_name': 'default',
                'gscam_config': ("nvarguscamerasrc sensor-id=" + left_camera +
                                 " ! video/x-raw(memory:NVMM), width=(int)" + width +
                                 ", height=(int)" + height + ", format=(string)" + format +
                                 ", framerate=(fraction)" + fps +
                                 " ! nvvidconv flip-method=6 ! video/x-raw, format=(string)BGRx ! videoconvert"),
                'camera_info_url': 'file://$(find my_stereo_camera)/config/calibration/uncalibrated_parameters.ini',
                'frame_id': '/left_frame',
                'sync_sink': False
            }],
            remappings=[('camera/image_raw', 'image_raw'), ('camera/camera_info', 'camera_info')]
        ),

        Node(
            package='gscam',
            executable='gscam',
            name='right_camera',
            namespace='right',
            parameters=[{
                'camera_name': 'default',
                'gscam_config': ("nvarguscamerasrc sensor-id=" + right_camera +
                                 " ! video/x-raw(memory:NVMM), width=(int)" + width +
                                 ", height=(int)" + height + ", format=(string)" + format +
                                 ", framerate=(fraction)" + fps +
                                 " ! nvvidconv flip-method=6 ! video/x-raw, format=(string)BGRx ! videoconvert"),
                'camera_info_url': 'file://$(find my_stereo_camera)/config/calibration/uncalibrated_parameters.ini',
                'frame_id': '/right_frame',
                'sync_sink': False
            }],
            remappings=[('camera/image_raw', 'image_raw'), ('camera/camera_info', 'camera_info')]
        )
    ])
