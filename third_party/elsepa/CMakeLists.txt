# ELSEPA (Salvat, Jablonski & Powell), needed by cstool for Mott cross sections.
#
# ELSEPA is distributed under the CPC Non-profit Use Licence, which does not
# allow redistribution. Its source is therefore not part of this repository;
# it is downloaded (and checksum-verified) at configure time, or taken from a
# local copy of the archive given by ELSEPA_ZIP. See README.md in this folder.

set(ELSEPA_URL "https://nebula-simulator.github.io/data/elsepa.zip"
	CACHE STRING "Download location of the ELSEPA archive")
set(ELSEPA_SHA256 "86cb42ada162498e0a2e886e5fd6c06195b583e820b52d7b8e4dff0b276b3342"
	CACHE STRING "Expected SHA-256 of the ELSEPA archive")
set(ELSEPA_ZIP "" CACHE FILEPATH
	"Local copy of the ELSEPA archive (skips the download if set)")

message(STATUS "ELSEPA is not distributed with this repository; it is obtained "
	"from ${ELSEPA_URL} under the CPC Non-profit Use Licence. "
	"Publications must cite F. Salvat, A. Jablonski, C.J. Powell, "
	"Comput. Phys. Commun. 165 (2005) 157, doi:10.1016/j.cpc.2004.09.006.")

# Obtain the archive
if(ELSEPA_ZIP)
	set(_archive "${ELSEPA_ZIP}")
	file(SHA256 "${_archive}" _hash)
	if(NOT _hash STREQUAL ELSEPA_SHA256)
		message(FATAL_ERROR "ELSEPA archive ${_archive} has SHA-256 ${_hash}, "
			"expected ${ELSEPA_SHA256}.")
	endif()
else()
	set(_archive "${CMAKE_CURRENT_BINARY_DIR}/elsepa.zip")
	if(EXISTS "${_archive}")
		file(SHA256 "${_archive}" _hash)
	endif()
	if(NOT EXISTS "${_archive}" OR NOT _hash STREQUAL ELSEPA_SHA256)
		message(STATUS "Downloading ELSEPA from ${ELSEPA_URL}")
		file(DOWNLOAD "${ELSEPA_URL}" "${_archive}"
			EXPECTED_HASH SHA256=${ELSEPA_SHA256}
			STATUS _status)
		list(GET _status 0 _code)
		if(NOT _code EQUAL 0)
			file(REMOVE "${_archive}")
			message(FATAL_ERROR "Could not download ELSEPA (${_status}). "
				"Download ${ELSEPA_URL} manually and pass -DELSEPA_ZIP=<file>, "
				"or disable it with -DNEBULA_BUILD_ELSEPA=OFF.")
		endif()
	endif()
endif()

# Unpack; the archive has a single top-level folder "elsepa"
set(_src "${CMAKE_CURRENT_BINARY_DIR}/src/elsepa")
if(NOT EXISTS "${_src}/elsepa.f")
	file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/src")
endif()

# ELSEPA reads its z_nnn.den and z_nnn.dfs data files from the working
# directory, and cstool runs it from a copy of the folder that contains the
# elscata executable. Put executables and data together in a dedicated folder.
set(ELSEPA_OUTPUT_DIR "${PROJECT_BINARY_DIR}/elsepa")
file(GLOB _data "${_src}/z_*.den" "${_src}/z_*.dfs")
file(COPY ${_data} "${_src}/readme.txt" DESTINATION "${ELSEPA_OUTPUT_DIR}")

foreach(_prog elscata elscatm)
	add_executable(${_prog} "${_src}/${_prog}.f")
	# elscat?.f INCLUDEs elsepa.f
	target_include_directories(${_prog} PRIVATE "${_src}")
	if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
		target_compile_options(${_prog} PRIVATE -std=legacy)
	endif()
	set_target_properties(${_prog} PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${ELSEPA_OUTPUT_DIR}")
endforeach()
