# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB MAIN_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h*)

add_demo(NAME interactive_face_detection_demo
    SOURCES ${MAIN_SRC}
    HEADERS ${MAIN_HEADERS}
    DEPENDENCIES monitors)
