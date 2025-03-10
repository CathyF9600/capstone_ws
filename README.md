# rob489-capstone
## Clone this repository
https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
## Setup Notes:
1. Wireless connection with jetson (ssh)
 - connect both laptop and jetson to TP-link course wifi
 - on laptop, run `ssh jetson@10.42.0.103` password is `jetson`
 - on flight, connect the green power board to the lipo battery
2. Wiring connecction with jetson (usage: code dev on jetson)
 - require: monitor, mouse, keyboard
 - connect jetson to a wifi with internet service
3. Jetson nano fan: in terminal run `fan_on` or `fan_off`
 - their command are encoded in ~/.bashrc: eg. `sudo sh -c 'sudo echo 255 > /sys/devices/pwm-fan/target_pwm'`
4. Build
 - `colcon build --symlink-install`
## Exercise 2:
1. Start MAVROS & PX4
    - `ros2 launch px4_autonomy_modules mavros.launch.py`
    - If not working correctly, `ls /dev/ttyUSB*`, expected to see `ttyUSB0`
2. Start RealSense Pose Estimation
    - `ros2 launch realsense2_camera rs_launch.py`
3. Run the Drone Communication Node
    - `ros2 run rob498_drone comm_node`
4. Check VICON & RealSense Pose Data (Optional)
   - `ros2 topic echo /vicon/ROB498_Drone/ROB498_Drone`
   - `ros2 topic echo /camera/pose/sample`
4. Send Commands Using ROS 2 Services (TA will run this during the actual test)
   - `ros2 service call /comm/launch std_srvs/srv/Trigger` (<-)
   - `ros2 service call /comm/test std_srvs/srv/Trigger` 
   - `ros2 service call /comm/land std_srvs/srv/Trigger` (<-)
   - `ros2 service call /comm/abort std_srvs/srv/Trigger`

## Installing PX4 Simulation on Ubuntu 20.04 arm64
- First upgrade your cmake to above 3.22 (https://answers.ros.org/question/293119/)
    - tar -xzvf cmake-3.30.8.tar.gz # source tar.gz (no distro)
    - cd ~/Downloads/cmake-3.30.8/   
    - ./bootstrap --prefix=$HOME/cmake-install
    - make -j$(nproc)
    - sudo make install
    - echo 'PATH=$HOME/cmake-install/bin:$PATH' >> ~/.bashrc
    - echo 'CMAKE_PREFIX_PATH=$HOME/cmake-install:$CMAKE_PREFIX_PATH' >> ~/.bashrc
    - NEVER RUN purge -autoremove cmake!!
- Setup PX4-Autopilot following (https://github.com/PX4/PX4-Autopilot/issues/21117)
- Setup Micro XRCE-DDS Agent & Client (https://github.com/PX4/PX4-user_guide/blob/main/tr/ros2/user_guide.md)




