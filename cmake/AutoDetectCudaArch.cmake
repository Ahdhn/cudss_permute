cmake_minimum_required(VERSION 3.23)

enable_language(CUDA)

if(NOT DEFINED CUDA_ARCHS)
	############################### Autodetect CUDA Arch #####################################################
	#Auto-detect cuda arch. Inspired by https://wagonhelm.github.io/articles/2018-03/detecting-cuda-capability-with-cmake
	# This will define and populates CUDA_ARCHS and put it in the cache 	
	set(cuda_arch_autodetect_file ${CMAKE_BINARY_DIR}/autodetect_cuda_archs.cu)		
	file(WRITE ${cuda_arch_autodetect_file} [[
		#include <stdio.h>
		int main() {
		int count = 0; 
		if (cudaSuccess != cudaGetDeviceCount(&count)) { return -1; }
		if (count == 0) { return -1; }
		for (int device = 0; device < count; ++device) {
			cudaDeviceProp prop; 
			bool is_unique = true; 
			if (cudaSuccess == cudaGetDeviceProperties(&prop, device)) {
				for (int device_1 = device - 1; device_1 >= 0; --device_1) {
					cudaDeviceProp prop_1; 
					if (cudaSuccess == cudaGetDeviceProperties(&prop_1, device_1)) {
						if (prop.major == prop_1.major && prop.minor == prop_1.minor) {
							is_unique = false; 
							break; 
						}
					}
					else { return -1; }
				}
				if (is_unique) {
					fprintf(stderr, "%d%d", prop.major, prop.minor);
				}
			}
			else { return -1; }
		}
		return 0; 
		}
		]])
	
	set(cuda_detect_cmd  "${CMAKE_CUDA_COMPILER} --run ${cuda_arch_autodetect_file}")
    message(STATUS "Executing: ${cuda_detect_cmd}")
	execute_process(COMMAND "${CMAKE_CUDA_COMPILER}" "--run" "${cuda_arch_autodetect_file}"
					#WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/CMakeFiles/"	
					RESULT_VARIABLE CUDA_RETURN_CODE	
					OUTPUT_VARIABLE dummy
					ERROR_VARIABLE fprintf_output					
					OUTPUT_STRIP_TRAILING_WHITESPACE
					ERROR_STRIP_TRAILING_WHITESPACE)							
	if(CUDA_RETURN_CODE EQUAL 0)
		# Clean the output to remove any warning messages and keep only numeric architecture values
		string(REGEX REPLACE "[^0-9]" "" clean_archs "${fprintf_output}")
		if(clean_archs STREQUAL "")
			message(STATUS "GPU architectures auto-detect failed (no valid architectures found). Will build for all possible architectures.")      
			set(CMAKE_CUDA_ARCHITECTURES all)
		else()
			# Convert string like "7586" to list like "75;86"
			string(LENGTH "${clean_archs}" arch_length)
			math(EXPR pairs "${arch_length} / 2")
			set(arch_list "")
			foreach(i RANGE 0 ${pairs})
				math(EXPR start_pos "${i} * 2")
				if(start_pos LESS arch_length)
					string(SUBSTRING "${clean_archs}" ${start_pos} 2 arch_pair)
					if(NOT arch_pair STREQUAL "")
						list(APPEND arch_list ${arch_pair})
					endif()
				endif()
			endforeach()
			set(CMAKE_CUDA_ARCHITECTURES "${arch_list}")
		endif()
	else()
		message(STATUS "GPU architectures auto-detect failed. Will build for all possible architectures.")      
		set(CMAKE_CUDA_ARCHITECTURES all)			
	endif()  	
	message(STATUS "CUDA architectures= " ${CMAKE_CUDA_ARCHITECTURES})	
endif()
###################################################################################
