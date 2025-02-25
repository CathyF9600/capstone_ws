# rob489-capstone
Setup Notes:
1. Wireless connection with jetson (ssh)
 - connect both laptop and jetson to TP-link course wifi
 - on laptop, run `ssh jetson@10.42.0.103` password is `jetson`
 - on flight, connect the green power board to the lipo battery
2. Wiring connecction with jetson (usage: code dev on jetson)
 - require: monitor, mouse, keyboard
 - connect jetson to a wifi with internet service
3. Jetson nano fan: in terminal run `fan_on` or `fan_off`
 - their command are encoded in ~/.bashrc: eg. `sudo sh -c 'sudo echo 255 > /sys/devices/pwm-fan/target_pwm'`

## Getting Realsense-Viewer to work:
- https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md
- https://unix.stackexchange.com/questions/399027/gpg-keyserver-receive-failed-server-indicated-a-failure
