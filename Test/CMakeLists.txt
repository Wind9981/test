cmake_minimum_required(VERSION 3.9)

set(PROJECT_NAME TestOpenCVCmake)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 11)


add_subdirectory(/home/nhan/dlib/dlib dlib_build)

add_subdirectory(/home/nhan/Ultra-Light-Fast-Generic-Face-Detector-1MB/ncnn/3rdparty/ncnn ncnn_build)
include_directories(/home/nhan/Ultra-Light-Fast-Generic-Face-Detector-1MB/ncnn/3rdparty/ncnn/src)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main test_utraFace.cpp UltraFace/UltraFace.cpp)

target_link_libraries(main ${OpenCV_LIBS} dlib::dlib ncnn)

add_executable(train_model train_model.cpp)

target_link_libraries(train_model ${OpenCV_LIBS} dlib::dlib stdc++fs)

add_executable(train_model2 train_model2.cpp UltraFace/UltraFace.cpp)

target_link_libraries(train_model2 ${OpenCV_LIBS} dlib::dlib stdc++fs ncnn)


add_executable(data_view face_data_webcam.cpp)

target_link_libraries(data_view dlib::dlib)

