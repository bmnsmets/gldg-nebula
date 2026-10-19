cmake_minimum_required(VERSION 3.18)
project(nebula C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

option(NEBULA_ENABLE_CUDA "Build the GPU version of Nebula if a CUDA compiler is found" ON)
option(NEBULA_BUILD_ELSEPA "Download and build ELSEPA, which cstool needs" ON)
option(NEBULA_PYTHON_ENV "Set up the project's Python environment (.venv, with cstool) using uv" ON)
set(NEBULA_MATERIALS "all" CACHE STRING
	"Materials to generate .mat files for: \"all\" (every .yaml in third_party/cstool/data/materials), a list such as \"silicon;pmma\", or empty for none")


# Find CUDA, if available. Besides $CUDACXX and PATH, which CMake checks
# itself, look for nvcc in $CUDA_HOME and the newest /usr/local/cuda*.
# Override with -DCMAKE_CUDA_COMPILER=/path/to/nvcc.
if(NEBULA_ENABLE_CUDA)
	if(NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
		file(GLOB _cuda_dirs /usr/local/cuda*)
		list(SORT _cuda_dirs COMPARE NATURAL ORDER DESCENDING)
		find_program(NEBULA_NVCC nvcc
			PATHS ENV CUDA_HOME ${_cuda_dirs}
			PATH_SUFFIXES bin)
		if(NEBULA_NVCC)
			set(CMAKE_CUDA_COMPILER "${NEBULA_NVCC}")
		endif()
	endif()
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
string(APPEND CMAKE_CUDA_FLAGS " -prec-sqrt=false -ftz=true -prec-div=false")

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


# Python environment of the project (pyproject.toml), with cstool, in
# <source>/.venv. Managed by uv.
set(_cstool "${PROJECT_SOURCE_DIR}/third_party/cstool")
set(_venv "${PROJECT_SOURCE_DIR}/.venv")
file(GLOB_RECURSE _cstool_sources CONFIGURE_DEPENDS
	"${_cstool}/cstool/*.py" "${_cstool}/apps/*.py")
if(NEBULA_PYTHON_ENV)
	find_program(NEBULA_UV uv)
	if(NOT NEBULA_UV)
		message(FATAL_ERROR "uv not found (https://docs.astral.sh/uv/). It is "
			"needed for the Python environment; disable that with -DNEBULA_PYTHON_ENV=OFF.")
	endif()
	set(_pyproject_files "${PROJECT_SOURCE_DIR}/pyproject.toml" "${_cstool}/pyproject.toml")
	if(EXISTS "${PROJECT_SOURCE_DIR}/uv.lock")
		list(APPEND _pyproject_files "${PROJECT_SOURCE_DIR}/uv.lock")
	endif()
	# cstool is installed in editable mode, so changes to its code need no
	# reinstall
	add_custom_command(OUTPUT "${PROJECT_BINARY_DIR}/python_env.stamp"
		COMMAND "${NEBULA_UV}" sync --project "${PROJECT_SOURCE_DIR}"
		COMMAND "${CMAKE_COMMAND}" -E touch "${PROJECT_BINARY_DIR}/python_env.stamp"
		DEPENDS ${_pyproject_files}
		COMMENT "Setting up Python environment in ${_venv}"
		VERBATIM)
	add_custom_target(python_env ALL DEPENDS "${PROJECT_BINARY_DIR}/python_env.stamp")
endif()


# Material files, generated with cstool
set(_materials "${NEBULA_MATERIALS}")
if(_materials STREQUAL "all")
	file(GLOB _yamls CONFIGURE_DEPENDS "${_cstool}/data/materials/*.yaml")
	set(_materials "")
	foreach(_yaml IN LISTS _yamls)
		get_filename_component(_mat "${_yaml}" NAME_WE)
		list(APPEND _materials "${_mat}")
	endforeach()
	if(NOT TARGET python_env OR NOT TARGET elscata)
		message(WARNING "Not generating material files: this needs the Python "
			"environment (NEBULA_PYTHON_ENV) and ELSEPA (NEBULA_BUILD_ELSEPA).")
		set(_materials "")
	endif()
elseif(_materials AND (NOT TARGET python_env OR NOT TARGET elscata))
	message(FATAL_ERROR "NEBULA_MATERIALS needs the Python environment "
		"(NEBULA_PYTHON_ENV) and ELSEPA (NEBULA_BUILD_ELSEPA).")
endif()
if(_materials)
	set(_mat_dir "${PROJECT_BINARY_DIR}/materials")
	file(MAKE_DIRECTORY "${_mat_dir}")
	set(_mat_files "")
	foreach(_mat IN LISTS _materials)
		set(_yaml "${_cstool}/data/materials/${_mat}.yaml")
		if(NOT EXISTS "${_yaml}")
			message(FATAL_ERROR "Unknown material ${_mat}: ${_yaml} does not exist.")
		endif()
		# cstool writes <name>.mat to the working directory
		add_custom_command(OUTPUT "${_mat_dir}/${_mat}.mat"
			COMMAND "${CMAKE_COMMAND}" -E env "ELSEPA_DIR=$<TARGET_FILE_DIR:elscata>"
				"${_venv}/bin/cstool" "${_yaml}"
			WORKING_DIRECTORY "${_mat_dir}"
			DEPENDS "${_yaml}" elscata "${PROJECT_BINARY_DIR}/python_env.stamp"
				${_cstool_sources}
			COMMENT "Generating ${_mat}.mat (this takes several minutes)"
			VERBATIM)
		list(APPEND _mat_files "${_mat_dir}/${_mat}.mat")
	endforeach()
	add_custom_target(materials ALL DEPENDS ${_mat_files})
endif()


# Script that puts Nebula, ELSEPA and the Python environment on the PATH
set(_env "# Source this file to put Nebula, ELSEPA and the project's Python tools on your PATH.\n")
string(APPEND _env "export PATH=\"${PROJECT_BINARY_DIR}/bin\${PATH:+:\$PATH}\"\n")
if(TARGET elscata)
	string(APPEND _env "export ELSEPA_DIR=\"$<TARGET_FILE_DIR:elscata>\"\n")
endif()
if(TARGET python_env)
	string(APPEND _env "export PATH=\"${_venv}/bin:\$PATH\"\n")
endif()
file(GENERATE OUTPUT "${PROJECT_BINARY_DIR}/nebula-env.sh" CONTENT "${_env}")
