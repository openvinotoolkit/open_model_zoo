# Copyright (C) 2021-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

add_demo(NAME mri_reconstruction_demo
         SOURCES ${SOURCES}
         HEADERS ${HEADERS})
