cmake_minimum_required(VERSION 3.21)
project(frame_detection_cpp)

set(CMAKE_CXX_STANDARD 14)

add_executable(frame_detection_cpp main.cpp)

set(OpenCV_DIR $ENV{BUILD_DIR_PATH})
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
