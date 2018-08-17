#include "ros/ros.h"
#include "geometry_msgs/Vector3.h"

int main (int argc, char** argv)
{
  // initiate ros nodes and 
  ros::init (argc, argv, "arduino_talker");
  ros::NodeHandle n;

  // initiate publisher
  ros::Publisher pub =
    n.advertise<geometry_msgs::Vector3>("arduino_talker", 500);

  ros::Rate loop_rate(10);
  
  int count = 0;
  int angle = 60;

  while(ros::ok())
  {
    // x is for servo and y is for throttle, z is placeholder
    geometry_msgs::Vector3 msg;
    //servo
    msg.x = angle++;
    if (angle > 130) {
      angle = 60;
    }
    // esc/ speed
    msg.y = 0;

    pub.publish(msg);

    ros::spinOnce();

    loop_rate.sleep();
    count++;
  }

  return 0;
}
