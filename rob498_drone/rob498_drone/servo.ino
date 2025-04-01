#include <Servo.h>

Servo myServo;  // Create a servo object
int servoPin = 9;  // Define the servo signal pin

void setup() {
    myServo.attach(servoPin);  // Attach the servo to the pin
}

void loop() {
    // Sweep from 0 to 180 degrees
    for (int angle = 0; angle <= 180; angle++) {
        myServo.write(angle);
        delay(5);  // Adjust speed of movement
    }
    
    // Sweep from 180 to 0 degrees
    for (int angle = 180; angle >= 0; angle--) {
        myServo.write(angle);
        delay(5);  // Adjust speed of movement
    }
}
