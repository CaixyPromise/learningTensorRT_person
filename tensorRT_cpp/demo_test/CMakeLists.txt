cmake_minimum_required(VERSION 3.0)

project(demo_test LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)

add_executable(letterbox letterbox.cpp)
target_include_directories(letterbox PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(letterbox PRIVATE ${OpenCV_LIBS})

add_executable(simple simple.cu)
