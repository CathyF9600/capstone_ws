from setuptools import setup
import os
from glob import glob

package_name = 'px4_autonomy_modules'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='PX4 Autonomy Modules for ROS2 integration',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Add any executable nodes here
        ],
    },
)
