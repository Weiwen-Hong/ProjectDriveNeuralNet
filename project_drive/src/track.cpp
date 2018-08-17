#include "ros/ros.h"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Pose.h"
#include "sensor_msgs/LaserScan.h"
#include <iostream>
#include <vector>

struct ZED_Pose;
struct PC;
void handleOdom(const nav_msgs::Odometry::ConstPtr& msg);
void updatePose(const nav_msgs::Odometry::ConstPtr& msg);
void updateLidar(const sensor_msgs::LaserScan::ConstPtr& msg);
void copyPose(ZED_Pose*, ZED_Pose*);

const bool PRINT = true;


/************** For handleOdom    ************/
std::vector<geometry_msgs::PoseStamped> list;
ros::Publisher pub;


/************** For updatePose ************/
ZED_Pose *lastPose, *currentPose;

/************** For updateLidar ************/
PC *pc;

struct ZED_Pose
{
  // for location
  float loc_x;
  float loc_y;
  float loc_z;

  // for orientation
  float ori_x;
  float ori_y;
  float ori_z;
};

// point cloud
struct PC
{
  float angle_min;
  float angle_max;
  float angle_increment;
  float time_increment;
  float scan_time;
  std::vector<float> ranges;
  std::vector<float> intensities;
  int numOfRanges;

};


int main(int argc, char** argv)
{
  ros::init(argc, argv, "track");

  ros::NodeHandle node;

  // initialize
  lastPose = new ZED_Pose;
  currentPose = new ZED_Pose;
  pc = new PC;

  // gather pose info
  ros::Subscriber poseSub = node.subscribe("/zed/odom", 200, updatePose);

  // gather lidar
  ros::Subscriber lidarSub = node.subscribe("/scan", 200, updateLidar);

  // ros::Subscriber sub = node.subscribe("/zed/odom", 200, handleOdom);
  // pub = node.advertise<nav_msgs::Path>("track", 200);
  ros::spin();

  return 0;
}



void updateLidar(const sensor_msgs::LaserScan::ConstPtr& msg)
{
  pc->angle_min = msg->angle_min;
  pc->angle_max = msg->angle_max;
  pc->angle_increment = msg->angle_increment;
  pc->time_increment = msg->time_increment;
  pc->scan_time = msg->scan_time;
  pc->ranges = msg->ranges;
  pc->intensities = msg->intensities;

  pc->numOfRanges = pc->ranges.size();

  if (PRINT)
  {
      printf(
        "angle_min: %f\n"
      "angle_max: %f\n"
      "angle_increment: %f\n"
      "time_increment: %f\n"
      "scan_time: %f\n"
      "ranges: ",
      pc->angle_min, pc->angle_max, pc->angle_increment, pc->time_increment,
      pc->scan_time
      );
    for (int i = 0; i < pc->numOfRanges; i++){
      std::cout << pc->ranges[i];
    }
    printf(
        "\n"
        "intensities: "
        );
    for (int i = 0; i < pc->intensities.size(); i++){
      std::cout << pc->intensities[i];
    }
    printf(
        "\n"
        "numOfRanges: %d\n\n\n\n\n",
        pc->numOfRanges
        );
  }
}



void updatePose(const nav_msgs::Odometry::ConstPtr& msg)
{
  copyPose(currentPose, lastPose);

  // update location
  currentPose->loc_x = msg->pose.pose.position.x;
  currentPose->loc_y = msg->pose.pose.position.y;
  currentPose->loc_z = msg->pose.pose.position.z;

  // update orientation
  currentPose->ori_x = msg->pose.pose.orientation.x;
  currentPose->ori_y = msg->pose.pose.orientation.y;
  currentPose->ori_z = msg->pose.pose.orientation.z;

  if (PRINT)
  {
    printf(
        "Position:\n"
        "   x:%f\n"
        "   y:%f\n"
        "   z:%f\n"
        "Orientation:\n"
        "   x:%f\n"
        "   y:%f\n"
        "   z:%f\n\n\n\n",
        currentPose->loc_x,
        currentPose->loc_y,
        currentPose->loc_z,
        currentPose->ori_x,
        currentPose->ori_y,
        currentPose->ori_z
        );
  }
}



/**
 *  deep copy from from to to.
 */
void copyPose(ZED_Pose* from, ZED_Pose* to)
{
  to->loc_x = from->loc_x;
  to->loc_y = from->loc_y;
  to->loc_z = from->loc_z;

  to->ori_x = from->ori_x;
  to->ori_y = from->ori_y;
  to->ori_z = from->ori_z;
}



/*
 *  Publish a path with all the Odometry poses
 */
void handleOdom(const nav_msgs::Odometry::ConstPtr& msg)
{
  // record the infomation from incoming msg
  geometry_msgs::PoseStamped pose;
  pose.pose = msg->pose.pose;
  pose.header = msg->header;

  // record the pose
  list.push_back(pose);

  // construct a new empty path
  nav_msgs::Path path;
  path.header = msg->header;
  path.poses.resize(list.size());

  // fill the path
  for (unsigned int i = 0; i < list.size(); i++)
  {
    path.poses[i] = list[i];
  }

  // publish the path if ROS is running
  if (ros::ok())
  {
    pub.publish(path);
    ros::spinOnce();
  }
}
