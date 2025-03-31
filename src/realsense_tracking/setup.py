from setuptools import setup

package_name = 'realsense_tracking'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/realsense_t265.launch.py', 'launch/stereo.launch.py', 'launch/gscam.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='Tracking node using Intel RealSense T265 in ROS 2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracking_node = realsense_tracking.tracking_node:main',
            't265_node = realsense_tracking.t265_node:main',
            '2dod_node = realsense_tracking.2dod_node:main',
            'imx219_node = realsense_tracking.imx219_node:main'
        ],
    },
)
