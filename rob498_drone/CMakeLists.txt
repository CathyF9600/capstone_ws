cmake_minimum_required(VERSION 3.5)
project(rob498_drone)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)

install(
  DIRECTORY rob498_drone
  DESTINATION lib/python3.8/site-packages
)

install(
  PROGRAMS
  rob498_drone/comm_node.py
  DESTINATION lib/rob498_drone
)

ament_package()
