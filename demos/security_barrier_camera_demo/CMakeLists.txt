# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file (GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

ie_add_sample(NAME security_barrier_camera_demo
              SOURCES ${SOURCES}
              HEADERS ${HEADERS}
              OPENCV_DEPENDENCIES core highgui videoio)
