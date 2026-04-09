cmake_minimum_required(VERSION 3.5)
project(automatic_calibration)

# Find PCL package
find_package(PCL REQUIRED COMPONENTS common io filters segmentation visualization)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# Include PCL headers
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})



add_executable(refine refinement.cpp)


add_executable(lidar lidar.cpp)


target_link_libraries(refine ${OpenCV_LIBS} ${PCL_LIBRARIES} libjsoncpp.a)


target_link_libraries(lidar PRIVATE ${PCL_LIBRARIES} yaml-cpp)
