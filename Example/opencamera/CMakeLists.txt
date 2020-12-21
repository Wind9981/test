cmake_minimum_required(VERSION 3.9)

set(PROJECT_NAME Test)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 11)


find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(camera_test camera.cpp)

target_link_libraries(camera_test ${OpenCV_LIBS})




