cmake_minimum_required(VERSION 2.8.3)
project(arduino_ambassador)

find_package( catkin REQUIRED
  COMPONENTS
    roscpp
    geometry_msgs
    std_msgs
  )


catkin_package()

include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(arduinoTalker src/arduinoTalker.cpp)
target_link_libraries(arduinoTalker ${catkin_LIBRARIES})
add_dependencies(arduinoTalker arduino_ambassadoe_generate_messages_cpp)
