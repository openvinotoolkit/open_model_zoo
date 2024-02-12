# Copyright (C) 2021-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT TARGET utils_gapi)
    message(WARNING "gaze_estimation_demo_gapi skipped. No suitable OpenCV version detected.")
    return()
endif()

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_demo(NAME gaze_estimation_demo_gapi
    SOURCES ${SOURCES}
    HEADERS ${HEADERS}
    INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
    DEPENDENCIES monitors utils_gapi
    OPENCV_VERSION_REQUIRED 4.5.3)
