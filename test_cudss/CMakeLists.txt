#CMake minimum requirement 
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

set(exec_name "test_cudss")

#GLOB source files 
file(GLOB_RECURSE src_files main.cu util.h)#<TODO> change this to add your files 

#Executable
add_executable(${exec_name} ${src_files})

#Libs linked to the executable
target_link_libraries( ${exec_name}   	
	$<BUILD_INTERFACE:developer_flags>
	metis	
)

#gtest_discover_tests(${exec_name})
