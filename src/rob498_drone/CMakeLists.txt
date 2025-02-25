cmake_minimum_required(VERSION 3.5)
project(rob498_drone)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Install Python scripts
install(PROGRAMS
  task_2.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install package.xml
install(FILES
  package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
