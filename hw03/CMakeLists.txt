cmake_minimum_required(VERSION 3.1)
project(hw03)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV CONFIG REQUIRED)

add_executable(hw03 main.cpp EigenFaceRecognizer.cpp EigenFaceRecognizer.h utils.h utils.cpp)
target_link_libraries(hw03 ${OpenCV_LIBS})