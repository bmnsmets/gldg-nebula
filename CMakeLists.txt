cmake_minimum_required(VERSION 3.18)
project(nebula C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

option(NEBULA_ENABLE_CUDA "Build the GPU version of Nebula if a CUDA compiler is found" ON)
option(NEBULA_BUILD_ELSEPA "Download and build ELSEPA, which cstool needs" ON)


# Find CUDA, if available
if(NEBULA_ENABLE_CUDA)
	include(CheckLanguage)
	check_language(CUDA)
endif()
if(NEBULA_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
	# Default set of GPU architectures: native code (SASS) for each major GPU
	# generation from Turing onwards, plus PTX for the newest one so that the
	# driver can JIT-compile for future GPUs. SASS also runs on GPUs with the
	# same major and a higher minor compute capability (e.g. 8.6 -> 8.7,
	# 12.0 -> 12.1). Entries not supported by the CUDA compiler are dropped.
	# Override with -DCMAKE_CUDA_ARCHITECTURES="80;90;120" (or "native").
	if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
		set(_wanted_archs 75 80 86 89 90 100 120)
		execute_process(COMMAND "${CMAKE_CUDA_COMPILER}" --list-gpu-arch
			OUTPUT_VARIABLE _nvcc_archs
			RESULT_VARIABLE _nvcc_result
			ERROR_QUIET)
		string(APPEND _nvcc_archs "\n")
		set(_archs "")
		set(_highest "")
		foreach(_arch IN LISTS _wanted_archs)
			if(NOT _nvcc_result EQUAL 0 OR _nvcc_archs MATCHES "compute_${_arch}[^0-9]")
				list(APPEND _archs "${_arch}-real")
				set(_highest "${_arch}")
			endif()
		endforeach()
		if(_highest)
			list(APPEND _archs "${_highest}-virtual")
		endif()
		set(CMAKE_CUDA_ARCHITECTURES "${_archs}" CACHE STRING
			"CUDA architectures to build for")
	endif()

	enable_language(CUDA)

	if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
		message(STATUS "Compiling without CUDA support: version 11.0 or higher required.")
		set(USE_CUDA OFF)
	else()
		set(USE_CUDA ON)
		message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

		set(CMAKE_CUDA_STANDARD 17)
		set(CMAKE_CUDA_STANDARD_REQUIRED ON)
		set(CMAKE_CUDA_EXTENSIONS OFF)
		add_definitions(-DCUDA_HEADERS_AVAILABLE=1)
		include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
	endif()
elseif(NOT NEBULA_ENABLE_CUDA)
	message(STATUS "Compiling without CUDA support: disabled by NEBULA_ENABLE_CUDA=OFF")
	set(USE_CUDA OFF)
else()
	message(STATUS "Compiling without CUDA support: no compiler found")
	set(USE_CUDA OFF)
endif()


# Find HDF5 library
find_package(HDF5 1.8.13 REQUIRED C HL)
include_directories(${HDF5_INCLUDE_DIRS})
add_definitions(${HDF5_DEFINITIONS})
list(APPEND LIBRARIES ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})

# Find MPI library if compiling against parallel HDF5
if(HDF5_IS_PARALLEL)
	find_package(MPI REQUIRED)
	include_directories(${MPI_CXX_INCLUDE_PATH})
	list(APPEND LIBRARIES ${MPI_CXX_LIBRARIES})
endif()

# Find threading library
find_package(Threads REQUIRED)


if(NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
		"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
		FORCE)
endif()

string(APPEND CMAKE_CUDA_FLAGS " --expt-relaxed-constexpr")
string(APPEND CMAKE_CUDA_FLAGS " -prec-sqrt=false")

add_subdirectory(source)


# ELSEPA, for cstool (requires a Fortran compiler)
if(NEBULA_BUILD_ELSEPA)
	include(CheckLanguage)
	check_language(Fortran)
	if(CMAKE_Fortran_COMPILER)
		enable_language(Fortran)
		add_subdirectory(third_party/elsepa)
	else()
		message(WARNING "No Fortran compiler found: not building ELSEPA. "
			"cstool needs ELSEPA to generate material files.")
	endif()
endif()
