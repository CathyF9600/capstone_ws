from setuptools import setup

package_name = 'rob498_sim'

setup(
    name=package_name,
    version='0.0.1',
    packages=['rob498_sim'],
    data_files=[
        ('share/ament_index/resource_index/ros2/share', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'rclpy', 'geometry_msgs'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='ROS2 package for simulating task2',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'task2_sim = rob498_sim.task2_sim:main',
        ],
    },
)

