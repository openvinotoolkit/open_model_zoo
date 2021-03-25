# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file (GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

ie_add_sample(NAME social_distance_demo
              SOURCES ${SOURCES}
              HEADERS ${HEADERS}
              INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
              DEPENDENCIES monitors
              OPENCV_DEPENDENCIES core highgui videoio)
