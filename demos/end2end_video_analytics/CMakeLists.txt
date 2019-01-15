# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 2.8)
project (end2end_video_analytics)

include_directories (${CMAKE_CURRENT_SOURCE_DIR}/common)

add_subdirectory(end2end_video_analytics_ie)
add_subdirectory(end2end_video_analytics_opencv)

