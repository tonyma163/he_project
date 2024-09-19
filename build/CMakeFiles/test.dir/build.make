# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tony/Github/project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tony/Github/project/build

# Include any dependencies generated for this target.
include CMakeFiles/test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test.dir/flags.make

CMakeFiles/test.dir/heaan.cu.o: CMakeFiles/test.dir/flags.make
CMakeFiles/test.dir/heaan.cu.o: CMakeFiles/test.dir/includes_CUDA.rsp
CMakeFiles/test.dir/heaan.cu.o: /home/tony/Github/project/heaan.cu
CMakeFiles/test.dir/heaan.cu.o: CMakeFiles/test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/tony/Github/project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/test.dir/heaan.cu.o"
	/usr/local/cuda-12.6/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/test.dir/heaan.cu.o -MF CMakeFiles/test.dir/heaan.cu.o.d -x cu -rdc=true -c /home/tony/Github/project/heaan.cu -o CMakeFiles/test.dir/heaan.cu.o

CMakeFiles/test.dir/heaan.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/test.dir/heaan.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/test.dir/heaan.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/test.dir/heaan.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target test
test_OBJECTS = \
"CMakeFiles/test.dir/heaan.cu.o"

# External object files for target test
test_EXTERNAL_OBJECTS =

CMakeFiles/test.dir/cmake_device_link.o: CMakeFiles/test.dir/heaan.cu.o
CMakeFiles/test.dir/cmake_device_link.o: CMakeFiles/test.dir/build.make
CMakeFiles/test.dir/cmake_device_link.o: /usr/local/lib/libPhantom.so
CMakeFiles/test.dir/cmake_device_link.o: /usr/local/lib/libHEaaN.so
CMakeFiles/test.dir/cmake_device_link.o: CMakeFiles/test.dir/deviceLinkLibs.rsp
CMakeFiles/test.dir/cmake_device_link.o: CMakeFiles/test.dir/deviceObjects1.rsp
CMakeFiles/test.dir/cmake_device_link.o: CMakeFiles/test.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/tony/Github/project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/test.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test.dir/build: CMakeFiles/test.dir/cmake_device_link.o
.PHONY : CMakeFiles/test.dir/build

# Object files for target test
test_OBJECTS = \
"CMakeFiles/test.dir/heaan.cu.o"

# External object files for target test
test_EXTERNAL_OBJECTS =

test: CMakeFiles/test.dir/heaan.cu.o
test: CMakeFiles/test.dir/build.make
test: /usr/local/lib/libPhantom.so
test: /usr/local/lib/libHEaaN.so
test: CMakeFiles/test.dir/cmake_device_link.o
test: CMakeFiles/test.dir/linkLibs.rsp
test: CMakeFiles/test.dir/objects1.rsp
test: CMakeFiles/test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/tony/Github/project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test.dir/build: test
.PHONY : CMakeFiles/test.dir/build

CMakeFiles/test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test.dir/clean

CMakeFiles/test.dir/depend:
	cd /home/tony/Github/project/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tony/Github/project /home/tony/Github/project /home/tony/Github/project/build /home/tony/Github/project/build /home/tony/Github/project/build/CMakeFiles/test.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/test.dir/depend

