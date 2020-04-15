# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file (GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file (GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

ie_add_sample(NAME classification_demo
              SOURCES ${SOURCES}
              HEADERS ${HEADERS}
              DEPENDENCIES monitors
              OPENCV_DEPENDENCIES core highgui)
