cmake_minimum_required(VERSION 3.21)
project(frame_detection_cpp)

set(CMAKE_CXX_STANDARD 14)

add_executable(frame_detection_cpp main.cpp src/frame_detector.cpp include/frame_detector.h include/settings.h)


# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV 3.4 REQUIRED PATHS PATHS "/usr/local/opt/opencv3")


# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})


# Link your application with OpenCV libraries
target_link_libraries(frame_detection_cpp ${OpenCV_LIBS})