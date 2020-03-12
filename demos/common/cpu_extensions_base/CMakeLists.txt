#
# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ===============================================================================
# Generated file for building library with user generated CPU extensions
#
# Contains implementation of the basic layer methods
#
# Refer to the section "Adding Your Own Kernels to the Inference Engine" in
# OpenVINO* documentation (either online or offline in
# <INSTALL_DIR>/deployment_tools/documentation/docs/index.html an then navigate
# to the corresponding section).
# ===============================================================================

set(CPU_EXTENSIONS_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")

macro(ie_add_cpu_extension)
    set(options EXCLUDE_CPPLINT)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES DEPENDENCIES)
    cmake_parse_arguments(IE_EXTENSION "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    set(TARGET_NAME ${IE_EXTENSION_NAME})

    get_filename_component(CMAKE_PREFIX_PATH "${InferenceEngine_DIR}" ABSOLUTE)

    message(STATUS "Looking for inference engine configuration file at: ${CMAKE_PREFIX_PATH}")
    find_package(InferenceEngine 1.0)

    if (NOT InferenceEngine_FOUND)
        message(FATAL_ERROR "Inference Engine was not found")
    endif()

    file(GLOB_RECURSE SRC ${CPU_EXTENSIONS_BASE_DIR}/*.cpp)
    file(GLOB_RECURSE HDR ${CPU_EXTENSIONS_BASE_DIR}/*.hpp)

    if(WIN32)
        add_definitions(-DIMPLEMENT_INFERENCE_ENGINE_API)
    endif()

    set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

    if ("${InferenceEngine_INCLUDE_DIRS}" STREQUAL "")
        message(FATAL_ERROR "InferenceEngine_INCLUDE_DIRS was not set!")
    endif()

    if ("${InferenceEngine_LIBRARIES}" STREQUAL "")
        message(FATAL_ERROR "InferenceEngine_LIBRARIES was not set!")
    endif()

    include_directories (PRIVATE
        ${CPU_EXTENSIONS_BASE_DIR}/common
        ${InferenceEngine_INCLUDE_DIRS}
    )

    #enable_omp()

    add_library(${TARGET_NAME} SHARED ${SRC} ${HDR} ${IE_EXTENSION_SOURCES})
    target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} ${IE_EXTENSION_DEPENDENCIES})
    target_include_directories(${TARGET_NAME} PUBLIC ${CPU_EXTENSIONS_BASE_DIR})
    set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})

    #set_target_cpu_flags(${TARGET_NAME})
endmacro()
