import RPi.GPIO as GPIO
import time

servo_pin = 18  # Replace with your actual GPIO pin number

GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)

pwm = GPIO.PWM(servo_pin, 50)  # 50 Hz frequency
pwm.start(0)

def set_angle(angle):
    duty = angle / 18 + 2
    GPIO.output(servo_pin, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(1)
    GPIO.output(servo_pin, False)
    pwm.ChangeDutyCycle(0)

# Example usage
set_angle(90)  # Move servo to 90 degrees
time.sleep(1)
set_angle(0)   # Move servo to 0 degrees

pwm.stop()
GPIO.cleanup()
