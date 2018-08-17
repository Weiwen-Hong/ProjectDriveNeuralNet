// ZED includes
#include <sl/Camera.hpp>

// OpenCV includes
#include <opencv2/opencv.hpp>

//may need to include?
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace sl;

cv::Mat slMat2cvMat(sl::Mat& input);
void printHelp();

int main(int argc, char **argv)
{
  //Set up the Zed Camera
  Camera zed;
  InitParameters init_params;
  // init_params.camera_resolution = RESOLUTION_HD72V0;//60fps
  init_params.camera_resolution = RESOLUTION_VGA;//60fps
  init_params.depth_mode = DEPTH_MODE_PERFORMANCE;
  init_params.coordinate_units = sl::UNIT_METER;

  //Open the camera
  ERROR_CODE err = zed.open(init_params);
  if(err != SUCCESS)
  {
    zed.close();
    return 1;
  }

  printHelp();

  //Set Camera runtime parameters
  RuntimeParameters runtime_parameters;
  runtime_parameters.sensing_mode = SENSING_MODE_STANDARD;

  //Enable positional tracking
  TrackingParameters tracking_parameters;
  err = zed.enableTracking(tracking_parameters);
  if(err != SUCCESS)
  {
    zed.close();
    return 2;
  }
  tracking_parameters.initial_world_transform = Transform::identity();
  tracking_parameters.enable_spatial_memory = true;

  //Get left camera
  Resolution image_size = zed.getResolution();
  sl::Mat image_zed(image_size, sl::MAT_TYPE_8U_C4);
  cv::Mat image_ocv = slMat2cvMat(image_zed);

  //Create OpenCV images to display (lower resolution to fit screen)
  cv::Size displaySize(600, 400);
  cv::Mat image_ocv_display(displaySize, CV_8UC4);
  cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);
  Camera::sticktoCPUCore(2);

  //variables
  Pose zed_pose;
  cv::Mat map = cv::Mat::zeros( 500, 500, CV_8UC1 );
  cv::Point prev;
  cv::Point curr;
  curr.x=250;
  curr.y=250;
  float translation_left_to_center = zed.getCameraInformation().calibration_parameters.T.x *0.5f;

  //Loop until 'q' is pressed.
  char key = ' ';
  while(key != 'q')
  {
    if (zed.grab(runtime_parameters) == SUCCESS)
    {
      TRACKING_STATE state = zed.getPosition(zed_pose, REFERENCE_FRAME_WORLD);
      zed.retrieveImage(image_zed, VIEW_LEFT);
      cv::resize(image_ocv, image_ocv_display, displaySize);

      cv::imshow("Camera", image_ocv_display);
      cv::imshow("Map", map);
      //Display translation and timestamp
      //printf("Translation: tx: %.3f, ty:  %.3f, tz:  %.3f, timestamp: %llu\n",
        //  zed_pose.getTranslation().tx, zed_pose.getTranslation().ty,
        //  zed_pose.getTranslation().tz, zed_pose.timestamp);
      // Display orientation quaternion
      //printf("Orientation: ox: %.3f, oy:  %.3f, oz:  %.3f, ow: %.3f\n",
        //  zed_pose.getOrientation().ox, zed_pose.getOrientation().oy,
        //  zed_pose.getOrientation().oz, zed_pose.getOrientation().ow);

      //transformPose(zed_pose.pose_data, translation_left_to_center);
      if(state == TRACKING_STATE_OK)
      {
        prev = curr;
        //sl::float4 quaternion = zed_pose.getOrientation();
        //printf("Quat: %.3f,%.3f,%.3f,%.3f\n",quaternion[0],quaternion[1],
          //quaternion[2],quaternion[3]);

        //sl::float3 rotation = zed_pose.getEulerAngles();
        //printf("Angles: %.3f,%.3f,%.3f\n",rotation[0],rotation[1],rotation[2]);

        sl::float3 translation = zed_pose.getTranslation();
        printf("Translation: %.3f,%.3f,%.3f\n",translation[0],translation[1],translation[2]);
        curr.x=250+10*translation[0];
        curr.y=250-10*translation[2];

        line(map, prev, curr, 127,2,8);
      }

      key = cv::waitKey(10);
    }
  }
  zed.close();
  return 0;
}

/**
* Conversion function between sl::Mat and cv::Mat
**/
cv::Mat slMat2cvMat(sl::Mat& input)
{
  //Mapping between MAT_TYPE and CV_TYPE
  int cv_type = -1;
  switch (input.getDataType())
  {
    case sl::MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
    case sl::MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
    case sl::MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
    case sl::MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
    case sl::MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
    case sl::MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
    case sl::MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
    case sl::MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
    default: break;
  }
  return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM_CPU));
}

void printHelp()
{
  std::cout << " Press 'q' to quit." << std::endl;
}
