cmake_minimum_required(VERSION 3.5)
project(rob498_sim)

# Set the policy to handle Python properly
#cmake_policy(SET CMP0148 NEW)

# Find necessary packages
#find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
#find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)

find_package(std_srvs REQUIRED)

# Explicitly find Python3
find_package(Python3 REQUIRED)

# Optional: Find other dependencies like Gazebo
#find_package(gazebo_ros_pkgs REQUIRED)

install(
  DIRECTORY rob498_sim/
  DESTINATION lib/python3.8/site-packages
)

install(
  PROGRAMS
  rob498_sim/task2_sim.py
  DESTINATION lib/rob498_sim
)

ament_package()
