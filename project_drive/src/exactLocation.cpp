#include "ros/ros.h"
#include "geometry_msgs/Vector3.h"

// This file calculates the exact map by adding the lidar data points with the translation from imu

/*
void 
int main (int argc, char** argv) {
	ros::init (argc, argv, "exact_location");
	ros::NodeHandle n;

	ros::Publisher pub = n.advertise<geometry_msgs::Vector3>("exact_location", 1000);

	ros::Rate loop_rate(10);

	while(ros::ok()) {
		geometry_msgs::Vector3 msg;

		msg.x*/
