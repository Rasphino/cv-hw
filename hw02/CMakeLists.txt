cmake_minimum_required(VERSION 3.1)
project(hw02)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV CONFIG REQUIRED)

add_executable(hw02 main.cpp)
target_link_libraries(hw02 ${OpenCV_LIBS})
