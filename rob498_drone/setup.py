from setuptools import setup

package_name = "rob498_drone"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="your_name",
    maintainer_email="your_email@example.com",
    description="ROS2 package for Flight Exercise #2",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "comm_node = rob498_drone.comm_node:main",
            "task3 = rob498_drone.task3:main"
        ],
    },
)
