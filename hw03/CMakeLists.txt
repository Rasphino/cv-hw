cmake_minimum_required(VERSION 3.1)
project(hw03)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV CONFIG REQUIRED)

add_executable(train train.cpp EigenFaceRecognizer.cpp EigenFaceRecognizer.h utils.h utils.cpp)
add_executable(predict predict.cpp EigenFaceRecognizer.cpp EigenFaceRecognizer.h utils.h utils.cpp)

target_link_libraries(train ${OpenCV_LIBS})
target_link_libraries(predict ${OpenCV_LIBS})