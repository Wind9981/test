cmake_minimum_required(VERSION 3.9)

set(PROJECT_NAME TestOpenCVCmake)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(dlib REQUIRED)
include_directories(${dlib_INCLUDE_DIRS})

add_executable(face_rec dnn_face_recognition_ex.cpp)
target_link_libraries(face_rec ${OpenCV_LIBS} ${OpenCV_LIBS} )


