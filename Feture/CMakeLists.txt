cmake_minimum_required(VERSION 3.9)

set(PROJECT_NAME TestCmake)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 11)


find_package(dlib REQUIRED)

include_directories(${dlib_INCLUDE_DIRS})

add_executable(main face_data_webcam.cpp)

target_link_libraries(main ${dlib_LIBS})


