# rob489-capstone
![](https://github.com/CathyF9600/capstone_ws/blob/main/capstone_clip4%20(1).gif)
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

## Exercise 3:
`ros2 launch px4_autonomy_modules mavros.launch.py`
1. realsense to mavros
`ros2 topic echo /mavros/vision_pose/pose`

2. vicon to mavros
`ros2 topic echo /mavros/local_position/pose`

3. start task3 node 
`ros2 run rob498_drone task3`

## Project
`ros2 launch realsense2_camera rs_launch.py`

`ros2 run realsense_tracking t265_node` --> RGB-D Map

`ros2 run rob498_drone project` --> Planner

### Point Cloud Reconstruction:
1. `ros2 bag record /depth_image /rgb_image /camera/pose/sample /vicon/ROB498_Drone/ROB498_Drone -o rosbag_1`
2. Replay the bag and run `ros2 run realsense_tracking save_bag` which generates a folder with *.npy files
3. Then you can visualize it by running `python3 src/scripts/planner_clean.py` 

### RBG Camera with Gripper Actuation (Plug in Arduino to Jetson)
`python3 src/scripts/inference_actuation.py` --> actuates the Arduino code

# Other notes (not using)
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

## Using image_pipeline
- `https://github.com/ros-perception/image_pipeline.git`
- `ros2 launch stereo_image_proc stereo_image_proc.launch.py namespace:=stereo`
- `ros2 run image_view stereo_view stereo:=/stereo image:=image_rect_color`
bug: terminate called after throwing an instance of 'rclcpp::exceptions::InvalidTopicNameError'
  what():  Invalid topic name: topic name must not contain repeated '/':
  '/stereo/left//image'

- `ros2 run image_view image_view image:=/stereo/left/image_rect_color`

## Install VS Code on Ubuntu 20.04 arm64
- `sudo add-apt-repository "deb [arch=arm64] https://packages.microsoft.com/repos/vscode stable main"`
- `sudo apt update` `sudo apt install code` `code`

## Course links
https://github.com/utiasSTARS/ROB498-support
https://github.com/utiasSTARS/ROB498-flight

if free space >= 32GB
https://github.com/AastaNV/JEP/blob/master/script/install_opencv4.6.0_Jetpack5.sh



`sudo cp /opt/nvidia/jetson-gpio/etc/99-gpio.rules /etc/udev/rules.d/`
`sudo udevadm control --reload-rules && sudo udevadm trigger`

K_left = [[286.1167907714844, 0.0, 421.62689208984375, 0.0],
          [0.0, 286.27880859375, 399.5252990722656, 0.0],
          [0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 1.0]
]

