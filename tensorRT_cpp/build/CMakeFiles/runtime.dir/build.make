# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/tsr_yoloPerson/tensorRT_cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/tsr_yoloPerson/tensorRT_cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/runtime.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/runtime.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/runtime.dir/flags.make

CMakeFiles/runtime.dir/runtime.cu.o: CMakeFiles/runtime.dir/flags.make
CMakeFiles/runtime.dir/runtime.cu.o: ../runtime.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/tsr_yoloPerson/tensorRT_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/runtime.dir/runtime.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /root/tsr_yoloPerson/tensorRT_cpp/runtime.cu -o CMakeFiles/runtime.dir/runtime.cu.o

CMakeFiles/runtime.dir/runtime.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/runtime.dir/runtime.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/runtime.dir/runtime.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/runtime.dir/runtime.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/runtime.dir/runtime.cu.o.requires:

.PHONY : CMakeFiles/runtime.dir/runtime.cu.o.requires

CMakeFiles/runtime.dir/runtime.cu.o.provides: CMakeFiles/runtime.dir/runtime.cu.o.requires
	$(MAKE) -f CMakeFiles/runtime.dir/build.make CMakeFiles/runtime.dir/runtime.cu.o.provides.build
.PHONY : CMakeFiles/runtime.dir/runtime.cu.o.provides

CMakeFiles/runtime.dir/runtime.cu.o.provides.build: CMakeFiles/runtime.dir/runtime.cu.o


CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o: CMakeFiles/runtime.dir/flags.make
CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o: /usr/src/tensorrt/samples/common/logger.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/tsr_yoloPerson/tensorRT_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o -c /usr/src/tensorrt/samples/common/logger.cpp

CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /usr/src/tensorrt/samples/common/logger.cpp > CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.i

CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /usr/src/tensorrt/samples/common/logger.cpp -o CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.s

CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o.requires:

.PHONY : CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o.requires

CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o.provides: CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o.requires
	$(MAKE) -f CMakeFiles/runtime.dir/build.make CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o.provides.build
.PHONY : CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o.provides

CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o.provides.build: CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o


# Object files for target runtime
runtime_OBJECTS = \
"CMakeFiles/runtime.dir/runtime.cu.o" \
"CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o"

# External object files for target runtime
runtime_EXTERNAL_OBJECTS =

CMakeFiles/runtime.dir/cmake_device_link.o: CMakeFiles/runtime.dir/runtime.cu.o
CMakeFiles/runtime.dir/cmake_device_link.o: CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o
CMakeFiles/runtime.dir/cmake_device_link.o: CMakeFiles/runtime.dir/build.make
CMakeFiles/runtime.dir/cmake_device_link.o: libyolo_plugin.so
CMakeFiles/runtime.dir/cmake_device_link.o: libyolo_utils.so
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libnvinfer.so
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libnvparsers.so
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libnvonnxparser.so
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
CMakeFiles/runtime.dir/cmake_device_link.o: CMakeFiles/runtime.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/tsr_yoloPerson/tensorRT_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA device code CMakeFiles/runtime.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/runtime.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/runtime.dir/build: CMakeFiles/runtime.dir/cmake_device_link.o

.PHONY : CMakeFiles/runtime.dir/build

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
runtime: /usr/lib/x86_64-linux-gnu/libnvinfer.so
runtime: /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so
runtime: /usr/lib/x86_64-linux-gnu/libnvparsers.so
runtime: /usr/lib/x86_64-linux-gnu/libnvonnxparser.so
runtime: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
runtime: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
runtime: CMakeFiles/runtime.dir/cmake_device_link.o
runtime: CMakeFiles/runtime.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/tsr_yoloPerson/tensorRT_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable runtime"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/runtime.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/runtime.dir/build: runtime

.PHONY : CMakeFiles/runtime.dir/build

CMakeFiles/runtime.dir/requires: CMakeFiles/runtime.dir/runtime.cu.o.requires
CMakeFiles/runtime.dir/requires: CMakeFiles/runtime.dir/usr/src/tensorrt/samples/common/logger.cpp.o.requires

.PHONY : CMakeFiles/runtime.dir/requires

CMakeFiles/runtime.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/runtime.dir/cmake_clean.cmake
.PHONY : CMakeFiles/runtime.dir/clean

CMakeFiles/runtime.dir/depend:
	cd /root/tsr_yoloPerson/tensorRT_cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/tsr_yoloPerson/tensorRT_cpp /root/tsr_yoloPerson/tensorRT_cpp /root/tsr_yoloPerson/tensorRT_cpp/build /root/tsr_yoloPerson/tensorRT_cpp/build /root/tsr_yoloPerson/tensorRT_cpp/build/CMakeFiles/runtime.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/runtime.dir/depend
