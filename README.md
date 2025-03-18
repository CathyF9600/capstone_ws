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
   - `ros2 service call rob498_drone_3/comm/launch std_srvs/srv/Trigger` (<-)
   - `ros2 service call rob498_drone_3/comm/test std_srvs/srv/Trigger` 
   - `ros2 service call rob498_drone_3/comm/land std_srvs/srv/Trigger` (<-)
   - `ros2 service call rob498_drone_3/comm/abort std_srvs/srv/Trigger`

## Get PX4-Autopilot Gazebo-Classic on Ubuntu 20.04 arm64 with ROS2 foxy
- (For Micro XRCE-DDS only) Upgrade your cmake to above 3.22 (https://answers.ros.org/question/293119/)
    - tar -xzvf cmake-3.30.8.tar.gz # source tar.gz (no distro)
    - cd ~/Downloads/cmake-3.30.8/   
    - ./bootstrap --prefix=$HOME/cmake-install
    - make -j$(nproc)
    - sudo make install
    - echo 'PATH=$HOME/cmake-install/bin:$PATH' >> ~/.bashrc
    - echo 'CMAKE_PREFIX_PATH=$HOME/cmake-install:$CMAKE_PREFIX_PATH' >> ~/.bashrc
    - NEVER RUN purge -autoremove cmake!!
- Upgrade your numpy to above 1.24 (it might affect ROS2 foxy packages but will
    - `sudo apt remove python3-numpy`
    - `pip3 install numpy`
    - `sudo apt install ros-foxy-desktop python3-argcomplete`
- Setup PX4-Autopilot following (https://github.com/PX4/PX4-Autopilot/issues/21117)
    - git clone https://github.com/PX4/PX4-Autopilot.git
    - cd PX4-Autopilot.git
    - git checkout v1.11.3
    - git submodule update --init --recursive
    - bash ./PX4-Autopilot/Tools/setup/ubuntu.sh --no-nuttx
    - make px4_sitl gazebo
- (Optional) Setup Micro XRCE-DDS Agent & Client (https://github.com/PX4/PX4-user_guide/blob/main/tr/ros2/user_guide.md)
    - you don't need the Micro XRCE-DDS Agent unless you are running ROS 2 on a microcontroller (using Micro-ROS)
![image](https://github.com/user-attachments/assets/fbd46d9b-0250-4550-bcc9-7ee5bf4b6224)


## Install VS Code on Ubuntu 20.04 arm64
- `sudo add-apt-repository "deb [arch=arm64] https://packages.microsoft.com/repos/vscode stable main"`
- `sudo apt update` `sudo apt install code` `code`


## Realsense Frame Offset
- transformation of -0.07 m in z direction for now