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
CMAKE_SOURCE_DIR = /home/nhan/ttttttt

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nhan/ttttttt/build

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/face_data_webcam.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/face_data_webcam.cpp.o: ../face_data_webcam.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nhan/ttttttt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/main.dir/face_data_webcam.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/face_data_webcam.cpp.o -c /home/nhan/ttttttt/face_data_webcam.cpp

CMakeFiles/main.dir/face_data_webcam.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/face_data_webcam.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nhan/ttttttt/face_data_webcam.cpp > CMakeFiles/main.dir/face_data_webcam.cpp.i

CMakeFiles/main.dir/face_data_webcam.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/face_data_webcam.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nhan/ttttttt/face_data_webcam.cpp -o CMakeFiles/main.dir/face_data_webcam.cpp.s

CMakeFiles/main.dir/face_data_webcam.cpp.o.requires:

.PHONY : CMakeFiles/main.dir/face_data_webcam.cpp.o.requires

CMakeFiles/main.dir/face_data_webcam.cpp.o.provides: CMakeFiles/main.dir/face_data_webcam.cpp.o.requires
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/face_data_webcam.cpp.o.provides.build
.PHONY : CMakeFiles/main.dir/face_data_webcam.cpp.o.provides

CMakeFiles/main.dir/face_data_webcam.cpp.o.provides.build: CMakeFiles/main.dir/face_data_webcam.cpp.o


# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/face_data_webcam.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

main: CMakeFiles/main.dir/face_data_webcam.cpp.o
main: CMakeFiles/main.dir/build.make
main: /usr/local/lib/libdlib.a
main: /usr/local/cuda/lib64/libcudart_static.a
main: /usr/lib/aarch64-linux-gnu/librt.so
main: /usr/lib/aarch64-linux-gnu/librt.so
main: /usr/lib/aarch64-linux-gnu/libSM.so
main: /usr/lib/aarch64-linux-gnu/libICE.so
main: /usr/lib/aarch64-linux-gnu/libX11.so
main: /usr/lib/aarch64-linux-gnu/libXext.so
main: /usr/lib/aarch64-linux-gnu/libpng.so
main: /usr/lib/aarch64-linux-gnu/libz.so
main: /usr/lib/aarch64-linux-gnu/libjpeg.so
main: /usr/lib/aarch64-linux-gnu/libcblas.so
main: /usr/lib/aarch64-linux-gnu/liblapack.so
main: /usr/lib/aarch64-linux-gnu/libcublas.so
main: /usr/lib/aarch64-linux-gnu/libcudnn.so
main: /usr/local/cuda/lib64/libcurand.so
main: /usr/local/cuda/lib64/libcusolver.so
main: /usr/local/cuda/lib64/libcudart.so
main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nhan/ttttttt/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: main

.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/requires: CMakeFiles/main.dir/face_data_webcam.cpp.o.requires

.PHONY : CMakeFiles/main.dir/requires

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend:
	cd /home/nhan/ttttttt/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nhan/ttttttt /home/nhan/ttttttt /home/nhan/ttttttt/build /home/nhan/ttttttt/build /home/nhan/ttttttt/build/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/main.dir/depend
