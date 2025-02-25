from setuptools import setup

package_name = 'rob498_drone'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools', 'rclpy', 'std_srvs', 'geometry_msgs'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    author='Your Name',
    author_email='your_email@example.com',
    description='A ROS 2 package for controlling a drone in Task 2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'task_2 = task_2:main',  # This points to the main function in task_2.py
        ],
    },
)
