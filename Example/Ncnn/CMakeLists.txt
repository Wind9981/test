cmake_minimum_required(VERSION 3.9)
#set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(PROJECT_NAME TestOpenCVCmake)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(/home/nhan/Example/Ultra-Light-Fast-Generic-Face-Detector-1MB/ncnn/3rdparty/ncnn ncnn_build)
include_directories(/home/nhan/Example/Ultra-Light-Fast-Generic-Face-Detector-1MB/ncnn/3rdparty/ncnn/src)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(testncnn TestNcnn.cpp UltraFace/UltraFace.cpp)

target_link_libraries(testncnn ${OpenCV_LIBS} ncnn)


