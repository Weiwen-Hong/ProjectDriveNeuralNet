#include <Servo.h>

const int ESC_SERVO_PIN = 3;// was 10
const int PIN_LED = 13;

const int SDA_PIN = 18;
const int SCL_PIN = 19;

const float NEUTRAL_THROTTLE = 90; // 60 to 102
const float NEUTRAL_STEERING_ANGLE = 90; // 50 to 130


int current_throttle = 255;

Servo steeringServo;
Servo electronicSpeedController;

void setup()
{
  //connect steering servo and esc
  electronicSpeedController.attach( ESC_SERVO_PIN );

  //set speeds to neutral
  electronicSpeedController.write( NEUTRAL_THROTTLE );

  Serial.begin(9600);

  delay(1000);
  Serial.println("Neutral is 90");
  Serial.println("S = speed, R = rotate, G = go");
}

int incomingByte = 127;
int number = 0;
boolean setCarSpeed = true; // true = speed, false = rotate

void loop()
{
  if (Serial.available() > 0)
  {
    incomingByte = Serial.read();
    if (incomingByte == 'S') //S  full throttle 255
    {
      current_throttle = 180;
      Serial.println("full throttle\n\n");
    }
    else if (incomingByte == 'R') //R  full reverse 0
    {
      current_throttle = 0;
      Serial.println("full reverse\n\n");
    }
    else if (incomingByte == 'G') //G    neutral 127
    {
      current_throttle = 90;
      Serial.println("neutral\n\n");
    }
  }
  //
  delay(20);
  electronicSpeedController.write(current_throttle);

  /*delay(5000);
    for(int i=0;i<20;i++)
    {
    electronicSpeedController.write( NEUTRAL_THROTTLE +i);
    Serial.println(NEUTRAL_THROTTLE + i);
    delay(2000);
    }
    electronicSpeedController.write( NEUTRAL_THROTTLE );
    delay(10000);
  */

  /*steeringServo.write( NEUTRAL_STEERING_ANGLE - 10);
    delay(2000);
    steeringServo.write( NEUTRAL_STEERING_ANGLE + 10);
    delay(2000);

    steeringServo.write( NEUTRAL_STEERING_ANGLE - 20);
    delay(2000);
    steeringServo.write( NEUTRAL_STEERING_ANGLE + 20);
    delay(2000);

    steeringServo.write( NEUTRAL_STEERING_ANGLE - 30);
    delay(2000);
    steeringServo.write( NEUTRAL_STEERING_ANGLE + 30);
    delay(2000);

    steeringServo.write( NEUTRAL_STEERING_ANGLE - 40);
    delay(2000);
    steeringServo.write( NEUTRAL_STEERING_ANGLE + 40);
    delay(2000);

    steeringServo.write(NEUTRAL_STEERING_ANGLE);
    delay(10000);*/

}
