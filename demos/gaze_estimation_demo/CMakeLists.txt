# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file (GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

ie_add_sample(NAME gaze_estimation_demo
              SOURCES ${SOURCES}
              HEADERS ${HEADERS}
              INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
              OPENCV_DEPENDENCIES highgui)        
