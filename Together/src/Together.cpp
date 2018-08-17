#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <cmath>

//Sweep includes
#include <sweep/sweep.hpp>

//Zed includes
#include <sl_zed/Camera.hpp>

//OpenCV includes
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace sl;
cv::Mat slMat2cvMat(Mat& input);
void gatherLIDAR(sweep::sweep& device);

bool lidarScan = true;
std::queue<sweep::scan> lidar;


int main(int argc, char **argv)
{
  if(argc !=2)
  {
    return 1;
  }
  try
  {
    //Set up the devices
    sweep::sweep device{argv[1]};
    std::thread scanse (gatherLIDAR,std::ref(device));
    
    Camera zed;
    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION_VGA;
    init_params.depth_mode = DEPTH_MODE_PERFORMANCE;
    init_params.coordinate_units = UNIT_METER;
    
    //Open the camera
    ERROR_CODE err = zed.open(init_params);
    if(err != SUCCESS)
    {
      printf("%s\n", toString(err).c_str());
      zed.close();
      return 1;
    }
    
    //Set Camera runtime parameters
    RuntimeParameters runtime_parameters;
    runtime_parameters.sensing_mode = SENSING_MODE_STANDARD;

    //Enable positional tracking
    TrackingParameters tracking_parameters;
    err = zed.enableTracking(tracking_parameters);
    if(err != SUCCESS)
    {
      printf("%s\n", toString(err).c_str());
      zed.close();
      return 2;
    }
    tracking_parameters.initial_world_transform = Transform::identity();
    tracking_parameters.enable_spatial_memory = true;
    
    int width = zed.getResolution().width;
    int height = zed.getResolution().height;
    Mat image_zed(width, height, MAT_TYPE_8U_C4);
    
    cv::Mat image_ocv = slMat2cvMat(image_zed);
    
    //variables
    Pose zed_pose;
    cv::Mat map = cv::Mat::zeros(500,500,CV_8UC1);
    cv::Point carPrev;
    cv::Point carCurr;
    carCurr.x = 250;
    carCurr.y = 250;
    
    char key = ' ';
    while(key != 'q')
    {
      if(zed.grab(runtime_parameters) == SUCCESS)
      {
        TRACKING_STATE state = zed.getPosition(zed_pose, REFERENCE_FRAME_WORLD);
        zed.retrieveImage(image_zed, VIEW_LEFT, MEM_CPU, width, height);
        if(state == TRACKING_STATE_OK)
        {
          carPrev = carCurr;
          sl::float3 translation = zed_pose.getTranslation();
          carCurr.x = 250+10*translation[0];
          carCurr.y = 250-10*translation[2];
          
          line(map, carPrev, carCurr, 127, 2, 8);
        }
        
        if(!lidar.empty())
        {
          map = cv::Mat::zeros(500,500,CV_8UC1);
          for(const sweep::sample& sample : lidar.front().samples)
          {
            cv::Point pt(carCurr.x+cos(sample.angle*M_PI/180000)*sample.distance,carCurr.y-sin(sample.angle*M_PI/180000)*sample.distance);
            line(map, pt, pt, 255, 2, 8);
            //line(map, pt, pt, sample.signal_strength, 2, 8);
          }
          lidar.pop();
        }
        
        cv::imshow("Camera", image_ocv);
        cv::imshow("Map", map);
        
        key = cv::waitKey(10);
      }
    }
    
    //close devices
    lidarScan = false;
    scanse.join();
    zed.close();
    return 0;
  }
  catch(const sweep::device_error& e)
  {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}

/**
* Conversion function between sl::Mat and cv::Mat
**/
cv::Mat slMat2cvMat(Mat& input)
{
  //Mapping between MAT_TYPE and CV_TYPE
  int cv_type = -1;
  switch (input.getDataType())
  {
    case MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
    case MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
    case MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
    case MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
    case MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
    case MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
    case MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
    case MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
    default: break;
  }
  return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM_CPU));
}

void gatherLIDAR(sweep::sweep& device)
{
  device.start_scanning();
  while(lidarScan)
  {
     lidar.push(device.get_scan());
  }
  device.stop_scanning();
}
