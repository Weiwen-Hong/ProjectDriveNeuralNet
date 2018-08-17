#include <Servo.h>

const int STEERING_SERVO_PIN = 2;
const int ESC_SERVO_PIN = 3;
const int PIN_LED = 13;
const int SDA_PIN = 18;
const int SCL_PIN = 19;

const int NEUTRAL_THROTTLE = 90; // 60 to 102
const int NEUTRAL_STEERING_ANGLE = 90; // 55 to 135

Servo steeringServo;
Servo electronicSpeedController;

void setup()
{
  Serial.begin(9600);

  //connect servo and esc
  steeringServo.attach( STEERING_SERVO_PIN );
  electronicSpeedController.attach( ESC_SERVO_PIN );

  //set speeds to neutral
  steeringServo.write( NEUTRAL_STEERING_ANGLE );
  steeringServo.write( NEUTRAL_THROTTLE );
}

char incomingByte;
char buffer[64];
int i = 0;
int space;
int speed = 0;
int rotate = 0;

void loop()//send '<speed> <rotate>\n'
{
  if(Serial.available() > 0)
  {
    incomingByte = Serial.read();
    if(incomingByte == '\n')
    {
      buffer[i] = 0;
      i = 0;
      speed = atol(buffer);
      rotate = atol(buffer+space);
//      Serial.print(speed);
//      Serial.print(' ');
//      Serial.println(rotate);
      electronicSpeedController.write(speed);
      steeringServo.write(rotate);
    }
    else if(incomingByte == ' ')
    {
      space = i+1;
      buffer[i++]=incomingByte;
    }
    else
    {
      buffer[i++] = incomingByte;
    }
  }
}
