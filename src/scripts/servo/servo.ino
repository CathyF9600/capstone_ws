#include <Servo.h>

Servo myServo;  // Create a servo object
int servoPin = 9;  // Define the servo signal pin

void setup() {
    Serial.begin(9600); // Initialize serial communication
    myServo.attach(servoPin);  // Attach the servo to the pin
}

void loop() {
    if (Serial.available() > 0) { // Check if data is available
        String command = Serial.readStringUntil('\n'); // Read the command
        command.trim(); // Remove any whitespace or newlines
        
        if (command == "ON") {
            for (int angle = 0; angle <= 180; angle++) {
                myServo.write(angle);
                delay(5);  // Adjust speed of movement
                Serial.println("ON_CONFIRM");
            }
        } else if (command == "OFF") {
            for (int angle = 180; angle >= 0; angle--) {
                myServo.write(angle);
                delay(5);  // Adjust speed of movement
                Serial.println("OFF_CONFIRM");
            }
        }
    }
}

