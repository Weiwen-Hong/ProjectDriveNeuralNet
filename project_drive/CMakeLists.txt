cmake_minimum_required(VERSION 2.8.3)
project(project_drive)

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
add_dependencies(arduinoTalker ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(track src/track.cpp)
target_link_libraries(track ${catkin_LIBRARIES})
add_dependencies(track ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
