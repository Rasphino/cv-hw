cmake_minimum_required(VERSION 3.1)
project(hw01)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV CONFIG REQUIRED)

add_executable(hw01 main.cpp)
target_link_libraries(hw01 ${OpenCV_LIBS})