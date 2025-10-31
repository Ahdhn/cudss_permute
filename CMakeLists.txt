#CMake minimum requirement 
#To run with cuDSS:
#>> cmake -DCMAKE_PREFIX_PATH="C:\Program Files\NVIDIA cuDSS\v0.6\lib\12\cmake" ..

cmake_minimum_required(VERSION 3.23 FATAL_ERROR)

#Project name 
set(project_name "TestcuDSS")#<TODO> change this to the name of your project 
project(${project_name} LANGUAGES CXX C CUDA)  

#toggle between building a shared or static library
option(MY_BUILD_SHARED_LIBS "Build as shared library" OFF)

#default build type is Release
if (CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Direct all output to /bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# Auto-detect GPU architecture
include("cmake/AutoDetectCudaArch.cmake")
include("cmake/metis.cmake")

# CUDA and C++ compiler flags
set(cxx_flags 
	$<$<CXX_COMPILER_ID:MSVC>:-D_SCL_SECURE_NO_WARNINGS /openmp /std:c++17> #<TODO> Add MSVC-specific compiler flags here
	$<$<CXX_COMPILER_ID:GNU>:-Wall -m64 -fopenmp -O3 -std=c++17>            #<TODO> Add GCC compiler flags here
	$<$<CXX_COMPILER_ID:Clang>:-Wall -m64 -fopenmp -O3 -std=c++17>          #<TODO> Add Clang compiler flags here
	)

set(MSVC_XCOMPILER_FLAGS "/openmp /std:c++17")
set(cuda_flags
    $<$<CXX_COMPILER_ID:GNU>:-Xcompiler -Wall -fopenmp -O3>
	$<$<CXX_COMPILER_ID:Clang>:-Xcompiler -Wall -fopenmp -O3>
	$<$<CXX_COMPILER_ID:MSVC>:-Xcompiler ${MSVC_XCOMPILER_FLAGS}>
    -Xcudafe=--display_error_number    
	-lineinfo	
	--expt-extended-lambda	
	-use_fast_math		
    $<$<CXX_COMPILER_ID:GNU>:-O3>	
	--expt-relaxed-constexpr	
    -Xptxas -warn-spills -res-usage   
	--ptxas-options=-v	
    #-G	
)
add_library(developer_flags INTERFACE)
target_compile_options(developer_flags INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
    $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>
)
target_include_directories(developer_flags INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_compile_features(developer_flags INTERFACE cxx_std_17)

#OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(developer_flags INTERFACE OpenMP::OpenMP_CXX)
endif()

#cudss
find_package(cudss)
message(STATUS "Found cuDSS version ${cudss_VERSION}")	
target_link_directories(developer_flags INTERFACE ${cudss_LIBRARY_DIR})

if (WIN32)
    target_include_directories(developer_flags INTERFACE ${cudss_INCLUDE_DIR})
    target_link_libraries(developer_flags INTERFACE cudss)
	
	#Copying cuDSS DLL to the binary directory
	add_custom_target(CopyCUDSSDLL)			
	foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
		add_custom_command(
			OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}/cudss64_0.dll"
			COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}"
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
				"${cudss_BINARY_DIR}/cudss64_0.dll"
				"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}/cudss64_0.dll"
			DEPENDS "${cudss_BINARY_DIR}/cudss64_0.dll"
			COMMENT "Copying cudss64_0.dll for ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG} configuration"
			VERBATIM
		)				
		add_custom_target("CopyCUDSSDLL${CONFIG}"
			DEPENDS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}/cudss64_0.dll")				
		add_dependencies(CopyCUDSSDLL "CopyCUDSSDLL${CONFIG}")				
	endforeach()
	
else()
    target_link_libraries(developer_flags INTERFACE cudss_static)
endif()


add_subdirectory(test_cudss)