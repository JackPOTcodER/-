cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)

project(GalaxyCameraTest)

include_directories(/opt/Galaxy/include)
include_directories(/usr/local/include/opencv4)

link_directories(/usr/local/lib)
link_directories(/opt/Galaxy/lib)

add_executable(GalaxyCameraTest get_information.cpp)

# find_package(OpenCV REQUIRED)
# target_link_libraries(CameraTest ${OpenCV_LIBRARIES})

find_package(spdlog REQUIRED)
target_link_libraries(GalaxyCameraTest
    opencv_core
    opencv_highgui
    opencv_videoio
    opencv_video
    opencv_imgproc
    opencv_imgcodecs
    opencv_calib3d
    spdlog::spdlog
    gxiapi    
)
























