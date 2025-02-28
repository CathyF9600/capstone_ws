install(
    DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/
)

install(PROGRAMS
    offboard_py/offb_node.py
    DESTINATION lib/${PROJECT_NAME}
)
