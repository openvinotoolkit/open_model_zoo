# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file (GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file (GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

message(STATUS ${HEADERS})
message(STATUS ${SOURCES})

ie_add_sample(NAME imagenet_classification_demo
              SOURCES ${SOURCES}
              HEADERS ${HEADERS}
              INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
              DEPENDENCIES format_reader
              OPENCV_DEPENDENCIES core highgui
              )
