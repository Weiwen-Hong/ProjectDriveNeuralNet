#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <cmath>
#include <python.h>

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

std::vector<int> decision;
int speed = 0;
int rotation = 0;

Py_Initialize();
PyObject *pName, *pModule, *pClass, *pInstance, *pResult;
pName = PyString_FromString("map");
pModule = PyImport_Import(pName);
Py_DECREF(pName);

// create an instance of navigation object from python code
pClass = PyObject_GetAttrString(pModule, "Navigation");
pInstance = PyInstance_New(pClass, NULL, NULL);


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
    
    int inputSize = 600;
    float * inputs = new float[inputSize];
    
    float sum1, sum2, sum3, sum4, sum5, sum6 = 0;
    float avg1, avg2, avg3, avg4, avg5, avg6 = 0; // avg distance of points within a sector
    int count1, count2, count3, count4, count5, count6 = 0; // count of points within 2m
    int total1, total2, total3, total4, total5, total6 = 0; // total count of points within a sector
    int close = 0;
    int lastAngle= 0;
    
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
            
            if (sample.angle < lastAngle) {
                // new loop begins
                avg1 = sum1 / total1;
                avg2 = sum2 / total2;
                avg3 = sum3 / total3;
                avg4 = sum4 / total4;
                avg5 = sum5 / total5;
                avg6 = sum6 / total6;
                
                // call python function
                pResult = PyObject_CallMethod(pInstance, "update", "(ii)", close, count1, count2,
                    count3, count4, count5, count6, avg1, avg2, avg3, avg4, avg5, avg6);
                	
                decision = listTupleToVector_Int(pResult);
                rotation = decision[0];
                speed = decision[1];
                
                count1, count2, count3, count4, count5, count6 = 0;
                total1, total2, total3, total4, total5, total6 = 0;
                sum1, sum2, sum3, sum4, sum5, sum6 = 0;
                avg1, avg2, avg3, avg4, avg5, avg6 = 0; 
                close = 0;
            }
            
            if (sample.angle < 30000) {
                updateParams(sample.distance, total1, count1, sum1);
            } else if (sample.angle < 60000) {
                updateParams(sample.distance, total2, count2, sum2);
            } else if (sample.angle < 90000) {
                updateParams(sample.distance, total3, count3, sum3);
            } else if (sample.angle < 120000) {
                updateParams(sample.distance, total4, count4, sum4);
            } else if (sample.angle < 150000) {
                updateParams(sample.distance, total5, count5, sum5);
            } else if (sample.angle < 180000) {
                updateParams(sample.distance, total6, count6, sum6);
            }
            if (sample.distance < 1000) {
                close++;
            }
            lastAngle = sample.angle;
        }
   
           // cv::Point pt(carCurr.x+cos(sample.angle*M_PI/180000)*sample.distance,carCurr.y-sin(sample.angle*M_PI/180000)*sample.distance);
            //line(map, pt, pt, 255, 2, 8);
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

void updateParams(int distance, int total, int count, float sum)
    if (distance > 800) {
        sum += 800;
    } else if (distance < 200) {
        sum += distance;
        count += 1;
    } else {
        sum += distance;
    }
        total += 1;
    }
}

std::vector<int> listTupleToVector_Int(PyObject* incoming) {
	std::vector<int> data;
	if (PyTuple_Check(incoming)) {
		for(Py_ssize_t i = 0; i < PyTuple_Size(incoming); i++) {
			PyObject *value = PyTuple_GetItem(incoming, i);
			data.push_back( PyFloat_AsDouble(value) );
		}
	} else {
		if (PyList_Check(incoming)) {
			for(Py_ssize_t i = 0; i < PyList_Size(incoming); i++) {
				PyObject *value = PyList_GetItem(incoming, i);
				data.push_back( PyFloat_AsDouble(value) );
			}
		} else {
			throw logic_error("Passed PyObject pointer was not a list or tuple!");
		}
	}
	return data;
}
