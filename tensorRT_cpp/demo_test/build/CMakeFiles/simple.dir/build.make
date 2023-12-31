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
CMAKE_SOURCE_DIR = /root/tensorRT_person/tensorRT_cpp/demo_test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/tensorRT_person/tensorRT_cpp/demo_test/build

# Include any dependencies generated for this target.
include CMakeFiles/simple.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/simple.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/simple.dir/flags.make

CMakeFiles/simple.dir/simple.cu.o: CMakeFiles/simple.dir/flags.make
CMakeFiles/simple.dir/simple.cu.o: ../simple.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/tensorRT_person/tensorRT_cpp/demo_test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/simple.dir/simple.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /root/tensorRT_person/tensorRT_cpp/demo_test/simple.cu -o CMakeFiles/simple.dir/simple.cu.o

CMakeFiles/simple.dir/simple.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/simple.dir/simple.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/simple.dir/simple.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/simple.dir/simple.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/simple.dir/simple.cu.o.requires:

.PHONY : CMakeFiles/simple.dir/simple.cu.o.requires

CMakeFiles/simple.dir/simple.cu.o.provides: CMakeFiles/simple.dir/simple.cu.o.requires
	$(MAKE) -f CMakeFiles/simple.dir/build.make CMakeFiles/simple.dir/simple.cu.o.provides.build
.PHONY : CMakeFiles/simple.dir/simple.cu.o.provides

CMakeFiles/simple.dir/simple.cu.o.provides.build: CMakeFiles/simple.dir/simple.cu.o


# Object files for target simple
simple_OBJECTS = \
"CMakeFiles/simple.dir/simple.cu.o"

# External object files for target simple
simple_EXTERNAL_OBJECTS =

CMakeFiles/simple.dir/cmake_device_link.o: CMakeFiles/simple.dir/simple.cu.o
CMakeFiles/simple.dir/cmake_device_link.o: CMakeFiles/simple.dir/build.make
CMakeFiles/simple.dir/cmake_device_link.o: CMakeFiles/simple.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/tensorRT_person/tensorRT_cpp/demo_test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/simple.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/simple.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/simple.dir/build: CMakeFiles/simple.dir/cmake_device_link.o

.PHONY : CMakeFiles/simple.dir/build

# Object files for target simple
simple_OBJECTS = \
"CMakeFiles/simple.dir/simple.cu.o"

# External object files for target simple
simple_EXTERNAL_OBJECTS =

simple: CMakeFiles/simple.dir/simple.cu.o
simple: CMakeFiles/simple.dir/build.make
simple: CMakeFiles/simple.dir/cmake_device_link.o
simple: CMakeFiles/simple.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/tensorRT_person/tensorRT_cpp/demo_test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable simple"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/simple.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/simple.dir/build: simple

.PHONY : CMakeFiles/simple.dir/build

CMakeFiles/simple.dir/requires: CMakeFiles/simple.dir/simple.cu.o.requires

.PHONY : CMakeFiles/simple.dir/requires

CMakeFiles/simple.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/simple.dir/cmake_clean.cmake
.PHONY : CMakeFiles/simple.dir/clean

CMakeFiles/simple.dir/depend:
	cd /root/tensorRT_person/tensorRT_cpp/demo_test/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/tensorRT_person/tensorRT_cpp/demo_test /root/tensorRT_person/tensorRT_cpp/demo_test /root/tensorRT_person/tensorRT_cpp/demo_test/build /root/tensorRT_person/tensorRT_cpp/demo_test/build /root/tensorRT_person/tensorRT_cpp/demo_test/build/CMakeFiles/simple.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/simple.dir/depend

