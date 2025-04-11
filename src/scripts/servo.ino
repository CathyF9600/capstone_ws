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
        
        if (command == "OPEN") {
            myServo.write(100);  // Move to 100 degrees
            Serial.println("OPEN_CONFIRM");
        } 
        else if (command == "CLOSE") {
            myServo.write(0);  // Move to 0 degrees
            Serial.println("CLOSE_CONFIRM");
        }
    }
}

