/*
 * rosserial Subscriber Example
 * Blinks an LED on callback
 */

#include <ros.h>
#include <geometry_msgs/Vector3.h>
#include <std_msgs/Int32.h>
#include <Servo.h>

#define SERVO_PIN 2
#define ESC_PIN 3

const float NEUTRAL_THROTTLE = 90; // 0 to 180
const float NEUTRAL_STEERING_ANGLE = 90; // 60 to 130

ros::NodeHandle nh;
std_msgs::Int32 servoValue;
ros::Publisher pub("arduino_msg", &servoValue);

Servo steeringServo;
Servo electronicSpeedController;


void updateMotor( const geometry_msgs::Vector3& command){
  int servo = command.x;
  int throttle = command.y;
  
  // constrain values into safe zone
  servo = min(servo, 130);
  servo = max(servo, 60);

  throttle = min(throttle, 180);
  throttle = max(throttle, 0);

  steeringServo.write(servo);
  electronicSpeedController.write(throttle);
}

ros::Subscriber<geometry_msgs::Vector3> sub("arduino_talker", &updateMotor );

void setup()
{
  //connect steering servo and esc
  steeringServo.attach( SERVO_PIN );
  electronicSpeedController.attach( ESC_PIN );

  //set speeds to neutral
  steeringServo.write( NEUTRAL_STEERING_ANGLE );
  electronicSpeedController.write( NEUTRAL_THROTTLE );
  
  nh.initNode();
  nh.subscribe(sub);
  nh.advertise(pub);
}

void loop()
{
  nh.spinOnce();
  delay(1);
}
