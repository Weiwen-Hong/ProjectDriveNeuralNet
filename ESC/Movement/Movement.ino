#include <Servo.h>

const int STEERING_SERVO_PIN = 2;
const int ESC_SERVO_PIN = 3;// was 10
const int PIN_LED = 13;

const int SDA_PIN = 18;
const int SCL_PIN = 19;

const float NEUTRAL_THROTTLE = 90; // 60 to 102
const float NEUTRAL_STEERING_ANGLE = 90; // 50 to 130

Servo steeringServo;
Servo electronicSpeedController;

void setup()
{
  //connect steering servo and esc
  steeringServo.attach( STEERING_SERVO_PIN );
  electronicSpeedController.attach( ESC_SERVO_PIN );

  //set speeds to neutral
  steeringServo.write( NEUTRAL_STEERING_ANGLE );
  electronicSpeedController.write( NEUTRAL_THROTTLE );

  Serial.begin(9600);
  
  delay(1000);
  Serial.println("Neutral is 90");
  Serial.println("S = speed, R = rotate, G = go");
}

int incomingByte = 0;
int number = 0;
boolean setCarSpeed = true; // true = speed, false = rotate

void loop()
{
  if(Serial.available() > 0)
  {
    incomingByte = Serial.read();
    if(incomingByte == 83)//S
    {
      setCarSpeed = true;
      number = 0;
    }
    else if(incomingByte == 82)//R
    {
      setCarSpeed = false;
      number = 0;
    }
    else if(incomingByte == 71)//G
    {
      Serial.print("-> ");
      if(setCarSpeed)
      {
        Serial.print("Setting car speed to ");
        electronicSpeedController.write(number);
      }
      else
      {
        Serial.print("Setting rotation to ");
        steeringServo.write(number);
      }
      Serial.println(number);
    }
    else
    {
      number = number*10 + (incomingByte - '0');
    }
  }
  delay(100);
  
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
