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
CMAKE_BINARY_DIR = /root/tsr_yoloPerson/tensorRT_cpp

# Include any dependencies generated for this target.
include CMakeFiles/yolo_utils.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/yolo_utils.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/yolo_utils.dir/flags.make

CMakeFiles/yolo_utils.dir/utils/postprocess.cpp.o: CMakeFiles/yolo_utils.dir/flags.make
CMakeFiles/yolo_utils.dir/utils/postprocess.cpp.o: utils/postprocess.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/tsr_yoloPerson/tensorRT_cpp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/yolo_utils.dir/utils/postprocess.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/yolo_utils.dir/utils/postprocess.cpp.o -c /root/tsr_yoloPerson/tensorRT_cpp/utils/postprocess.cpp

CMakeFiles/yolo_utils.dir/utils/postprocess.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolo_utils.dir/utils/postprocess.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/tsr_yoloPerson/tensorRT_cpp/utils/postprocess.cpp > CMakeFiles/yolo_utils.dir/utils/postprocess.cpp.i

CMakeFiles/yolo_utils.dir/utils/postprocess.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolo_utils.dir/utils/postprocess.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/tsr_yoloPerson/tensorRT_cpp/utils/postprocess.cpp -o CMakeFiles/yolo_utils.dir/utils/postprocess.cpp.s

CMakeFiles/yolo_utils.dir/utils/postprocess.cpp.o.requires:

.PHONY : CMakeFiles/yolo_utils.dir/utils/postprocess.cpp.o.requires

CMakeFiles/yolo_utils.dir/utils/postprocess.cpp.o.provides: CMakeFiles/yolo_utils.dir/utils/postprocess.cpp.o.requires
	$(MAKE) -f CMakeFiles/yolo_utils.dir/build.make CMakeFiles/yolo_utils.dir/utils/postprocess.cpp.o.provides.build
.PHONY : CMakeFiles/yolo_utils.dir/utils/postprocess.cpp.o.provides

CMakeFiles/yolo_utils.dir/utils/postprocess.cpp.o.provides.build: CMakeFiles/yolo_utils.dir/utils/postprocess.cpp.o


CMakeFiles/yolo_utils.dir/utils/preprocess.cu.o: CMakeFiles/yolo_utils.dir/flags.make
CMakeFiles/yolo_utils.dir/utils/preprocess.cu.o: utils/preprocess.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/tsr_yoloPerson/tensorRT_cpp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/yolo_utils.dir/utils/preprocess.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /root/tsr_yoloPerson/tensorRT_cpp/utils/preprocess.cu -o CMakeFiles/yolo_utils.dir/utils/preprocess.cu.o

CMakeFiles/yolo_utils.dir/utils/preprocess.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/yolo_utils.dir/utils/preprocess.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/yolo_utils.dir/utils/preprocess.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/yolo_utils.dir/utils/preprocess.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/yolo_utils.dir/utils/preprocess.cu.o.requires:

.PHONY : CMakeFiles/yolo_utils.dir/utils/preprocess.cu.o.requires

CMakeFiles/yolo_utils.dir/utils/preprocess.cu.o.provides: CMakeFiles/yolo_utils.dir/utils/preprocess.cu.o.requires
	$(MAKE) -f CMakeFiles/yolo_utils.dir/build.make CMakeFiles/yolo_utils.dir/utils/preprocess.cu.o.provides.build
.PHONY : CMakeFiles/yolo_utils.dir/utils/preprocess.cu.o.provides

CMakeFiles/yolo_utils.dir/utils/preprocess.cu.o.provides.build: CMakeFiles/yolo_utils.dir/utils/preprocess.cu.o


# Object files for target yolo_utils
yolo_utils_OBJECTS = \
"CMakeFiles/yolo_utils.dir/utils/postprocess.cpp.o" \
"CMakeFiles/yolo_utils.dir/utils/preprocess.cu.o"

# External object files for target yolo_utils
yolo_utils_EXTERNAL_OBJECTS =

CMakeFiles/yolo_utils.dir/cmake_device_link.o: CMakeFiles/yolo_utils.dir/utils/postprocess.cpp.o
CMakeFiles/yolo_utils.dir/cmake_device_link.o: CMakeFiles/yolo_utils.dir/utils/preprocess.cu.o
CMakeFiles/yolo_utils.dir/cmake_device_link.o: CMakeFiles/yolo_utils.dir/build.make
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
CMakeFiles/yolo_utils.dir/cmake_device_link.o: CMakeFiles/yolo_utils.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/tsr_yoloPerson/tensorRT_cpp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA device code CMakeFiles/yolo_utils.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/yolo_utils.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/yolo_utils.dir/build: CMakeFiles/yolo_utils.dir/cmake_device_link.o

.PHONY : CMakeFiles/yolo_utils.dir/build

# Object files for target yolo_utils
yolo_utils_OBJECTS = \
"CMakeFiles/yolo_utils.dir/utils/postprocess.cpp.o" \
"CMakeFiles/yolo_utils.dir/utils/preprocess.cu.o"

# External object files for target yolo_utils
yolo_utils_EXTERNAL_OBJECTS =

libyolo_utils.so: CMakeFiles/yolo_utils.dir/utils/postprocess.cpp.o
libyolo_utils.so: CMakeFiles/yolo_utils.dir/utils/preprocess.cu.o
libyolo_utils.so: CMakeFiles/yolo_utils.dir/build.make
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
libyolo_utils.so: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
libyolo_utils.so: CMakeFiles/yolo_utils.dir/cmake_device_link.o
libyolo_utils.so: CMakeFiles/yolo_utils.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/tsr_yoloPerson/tensorRT_cpp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared library libyolo_utils.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/yolo_utils.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/yolo_utils.dir/build: libyolo_utils.so

.PHONY : CMakeFiles/yolo_utils.dir/build

CMakeFiles/yolo_utils.dir/requires: CMakeFiles/yolo_utils.dir/utils/postprocess.cpp.o.requires
CMakeFiles/yolo_utils.dir/requires: CMakeFiles/yolo_utils.dir/utils/preprocess.cu.o.requires

.PHONY : CMakeFiles/yolo_utils.dir/requires

CMakeFiles/yolo_utils.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/yolo_utils.dir/cmake_clean.cmake
.PHONY : CMakeFiles/yolo_utils.dir/clean

CMakeFiles/yolo_utils.dir/depend:
	cd /root/tsr_yoloPerson/tensorRT_cpp && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/tsr_yoloPerson/tensorRT_cpp /root/tsr_yoloPerson/tensorRT_cpp /root/tsr_yoloPerson/tensorRT_cpp /root/tsr_yoloPerson/tensorRT_cpp /root/tsr_yoloPerson/tensorRT_cpp/CMakeFiles/yolo_utils.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/yolo_utils.dir/depend
