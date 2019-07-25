# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME super_resolution_demo
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/super_resolution_demo.h"
              DEPENDENCIES format_reader
              OPENCV_DEPENDENCIES highgui)
