#include <Servo.h>

Servo myServo;
int servoPin = 9;

int currentAngle = 0;         // Start at 0 degrees
int stepDelay = 10;           // Delay between each step in ms (controls speed)
int stepSize = 1;             // Degrees per step

void setup() {
    Serial.begin(9600);
    myServo.attach(servoPin);
    myServo.write(currentAngle);
    Serial.println("READY");
}

void loop() {
    if (Serial.available() > 0) {
        String input = Serial.readStringUntil('\n');
        input.trim();

        if (input == "OPEN") {
            moveServoSmooth(100);
            Serial.println("OPEN_CONFIRM");
        } 
        else if (input == "CLOSE") {
            moveServoSmooth(0);
            Serial.println("CLOSE_CONFIRM");
        } 
        else if (input == "MID") {
            moveServoSmooth(50);
            Serial.println("MID_CONFIRM");
        } 
        else if (input.startsWith("SET")) {
            int angle = input.substring(3).toInt();
            angle = constrain(angle, 0, 180);  // Clamp between 0-180
            moveServoSmooth(angle);
            Serial.print("SET_CONFIRM ");
            Serial.println(angle);
        }
        else if (input == "STATUS") {
            Serial.print("ANGLE ");
            Serial.println(currentAngle);
        }
    }
}

// Smooth movement function
void moveServoSmooth(int targetAngle) {
    if (targetAngle == currentAngle) return;

    int step = (targetAngle > currentAngle) ? stepSize : -stepSize;
    for (int angle = currentAngle; angle != targetAngle; angle += step) {
        myServo.write(angle);
        delay(stepDelay);
    }
    myServo.write(targetAngle);  // Final adjustment
    currentAngle = targetAngle;
}
