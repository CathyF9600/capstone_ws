import Jetson.GPIO as GPIO
import time

servo_pin = 18  # Use GPIO18 (Pin 12)

GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
GPIO.setup(servo_pin, GPIO.OUT)

pwm = GPIO.PWM(servo_pin, 50)  # 50 Hz frequency
pwm.start(7.5)  # 7.5% duty cycle for neutral position

def set_angle(angle):
    duty_cycle = (angle / 18) + 2.5
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)

try:
    while True:
        set_angle(0)   # Move to 0 degrees
        time.sleep(1)
        set_angle(90)  # Move to 90 degrees
        time.sleep(1)
        set_angle(180) # Move to 180 degrees
        time.sleep(1)
except KeyboardInterrupt:
    pwm.stop()
    GPIO.cleanup()
