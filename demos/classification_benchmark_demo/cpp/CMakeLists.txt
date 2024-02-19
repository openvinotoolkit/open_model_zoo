# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_demo(NAME classification_benchmark_demo
    SOURCES ${SOURCES}
    HEADERS ${HEADERS}
    DEPENDENCIES monitors models pipelines)
