#include <Servo.h>

Servo myServo;
int servoPin = 10;
int currentAngle = 90; // Start at the middle position

void setup() {
    Serial.begin(9600);
    myServo.attach(servoPin);
    myServo.write(currentAngle); // Move to starting position
}

void loop() {
    if (Serial.available() > 0) {
        String command = Serial.readStringUntil('\n');
        command.trim();

        if (command == "ON") {
            //currentAngle = constrain(currentAngle - 5, 0, 180);
            myServo.write(45);
            Serial.println("ON_CONFIRM");
        } else if (command == "OFF") {
            //currentAngle = constrain(currentAngle + 5, 0, 180);
            myServo.write(0);
            Serial.println("OFF_CONFIRM");
        }
        delay(200);  // Optional: Slow down command response
    }
}
