# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# compile CUDA with /usr/local/cuda/bin/nvcc
# compile CXX with /usr/bin/c++
CUDA_DEFINES = -Dyolo_utils_EXPORTS

CUDA_INCLUDES = --options-file CMakeFiles/yolo_utils.dir/includes_CUDA.rsp

CUDA_FLAGS = -std=c++14 --generate-code=arch=compute_61,code=[compute_61,sm_61] --generate-code=arch=compute_70,code=[compute_70,sm_70] --generate-code=arch=compute_75,code=[compute_75,sm_75] -Xcompiler=-fPIC

CXX_DEFINES = -Dyolo_utils_EXPORTS

CXX_INCLUDES = -I/usr/local/cuda/include -isystem /usr/include/opencv4

CXX_FLAGS = -std=gnu++14 -fPIC

