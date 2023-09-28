# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /root/miniconda3/lib/python3.8/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /root/miniconda3/lib/python3.8/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/learning-tensor-rt_person/tensorRT_cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/learning-tensor-rt_person/tensorRT_cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/runtime.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/runtime.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/runtime.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/runtime.dir/flags.make

CMakeFiles/runtime.dir/runtime.cu.o: CMakeFiles/runtime.dir/flags.make
CMakeFiles/runtime.dir/runtime.cu.o: CMakeFiles/runtime.dir/includes_CUDA.rsp
CMakeFiles/runtime.dir/runtime.cu.o: /root/learning-tensor-rt_person/tensorRT_cpp/runtime.cu
CMakeFiles/runtime.dir/runtime.cu.o: CMakeFiles/runtime.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/learning-tensor-rt_person/tensorRT_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/runtime.dir/runtime.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/runtime.dir/runtime.cu.o -MF CMakeFiles/runtime.dir/runtime.cu.o.d -x cu -c /root/learning-tensor-rt_person/tensorRT_cpp/runtime.cu -o CMakeFiles/runtime.dir/runtime.cu.o

CMakeFiles/runtime.dir/runtime.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/runtime.dir/runtime.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/runtime.dir/runtime.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/runtime.dir/runtime.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o: CMakeFiles/runtime.dir/flags.make
CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o: /usr/src/tensorrt/samples/common/logger.cpp
CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o: CMakeFiles/runtime.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/learning-tensor-rt_person/tensorRT_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o -MF CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o.d -o CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o -c /usr/src/tensorrt/samples/common/logger.cpp

CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /usr/src/tensorrt/samples/common/logger.cpp > CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.i

CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /usr/src/tensorrt/samples/common/logger.cpp -o CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.s

# Object files for target runtime
runtime_OBJECTS = \
"CMakeFiles/runtime.dir/runtime.cu.o" \
"CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o"

# External object files for target runtime
runtime_EXTERNAL_OBJECTS =

runtime: CMakeFiles/runtime.dir/runtime.cu.o
runtime: CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o
runtime: CMakeFiles/runtime.dir/build.make
runtime: libyolo_plugin.so
runtime: libyolo_utils.so
runtime: streamer/streamer/libstreamer.a
runtime: libtask.so
runtime: /usr/lib/x86_64-linux-gnu/libnvinfer.so
runtime: /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so
runtime: /usr/lib/x86_64-linux-gnu/libnvparsers.so
runtime: /usr/lib/x86_64-linux-gnu/libnvonnxparser.so
runtime: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.2.0
runtime: /usr/lib/x86_64-linux-gnu/libavcodec.so
runtime: /usr/lib/x86_64-linux-gnu/libavformat.so
runtime: /usr/lib/x86_64-linux-gnu/libavutil.so
runtime: /usr/lib/x86_64-linux-gnu/libswscale.so
runtime: CMakeFiles/runtime.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/learning-tensor-rt_person/tensorRT_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable runtime"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/runtime.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/runtime.dir/build: runtime
.PHONY : CMakeFiles/runtime.dir/build

CMakeFiles/runtime.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/runtime.dir/cmake_clean.cmake
.PHONY : CMakeFiles/runtime.dir/clean

CMakeFiles/runtime.dir/depend:
	cd /root/learning-tensor-rt_person/tensorRT_cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/learning-tensor-rt_person/tensorRT_cpp /root/learning-tensor-rt_person/tensorRT_cpp /root/learning-tensor-rt_person/tensorRT_cpp/build /root/learning-tensor-rt_person/tensorRT_cpp/build /root/learning-tensor-rt_person/tensorRT_cpp/build/CMakeFiles/runtime.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/runtime.dir/depend

