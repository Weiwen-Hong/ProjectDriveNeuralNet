// ZED includes
#include <sl_zed/Camera.hpp>

// OpenCV includes
#include <opencv2/opencv.hpp>

//may need to include?
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace sl;

cv::Mat slMat2cvMat(Mat& input);
void printHelp();

int main(int argc, char **argv)
{
  //Set up the Zed Camera
  Camera zed;
  InitParameters init_params;
  init_params.camera_resolution = RESOLUTION_HD720;//60fps
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

  printHelp();

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

  //Prepare new image size to retrieve half-resolution images
  Resolution image_size = zed.getResolution();
  int new_width = image_size.width / 2;
  int new_height = image_size.height / 2;

  //Get left camera
  Mat image_zed(new_width, new_height, MAT_TYPE_8U_C4);
  cv::Mat image_ocv = slMat2cvMat(image_zed);

  //variables
  Pose zed_pose;

  //Loop until 'q' is pressed.
  char key = ' ';
  while(key != 'q')
  {
    if (zed.grab(runtime_parameters) == SUCCESS)
    {
      TRACKING_STATE state = zed.getPosition(zed_pose, REFERENCE_FRAME_WORLD);
      zed.retrieveImage(image_zed, VIEW_LEFT, MEM_CPU, new_width, new_height);
      //Display translation and timestamp
      printf("Translation: tx: %.3f, ty:  %.3f, tz:  %.3f, timestamp: %llu\n",
          zed_pose.getTranslation().tx, zed_pose.getTranslation().ty, zed_pose.getTranslation().tz, zed_pose.timestamp);
      // Display orientation quaternion
      printf("Orientation: ox: %.3f, oy:  %.3f, oz:  %.3f, ow: %.3f\n",
          zed_pose.getOrientation().ox, zed_pose.getOrientation().oy, zed_pose.getOrientation().oz, zed_pose.getOrientation().ow);

      cv::imshow("Camera", image_ocv);

      key = cv::waitKey(10);
    }
  }
  zed.close();
  return 0;
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

void printHelp()
{
  std::cout << " Press 'q' to quit." << std::endl;
}
