# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED CMakeScripts_DIR)
    message(FATAL_ERROR "CMakeScripts_DIR is not defined")
endif()

set(OLD_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "${CMakeScripts_DIR}")

# Code style utils
include(cpplint/cpplint)
include(clang_format/clang_format)