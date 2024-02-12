# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT TARGET utils_gapi)
    message(WARNING "classification_benchmark_demo skipped. No suitable OpenCV version detected.")
    return()
endif()

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_demo(NAME classification_benchmark_demo_gapi
    SOURCES ${SOURCES}
    HEADERS ${HEADERS}
    INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
    DEPENDENCIES monitors utils_gapi models pipelines
    OPENCV_VERSION_REQUIRED 4.5.2)
