# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME crossroad_camera_demo
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/crossroad_camera_demo.hpp"
              DEPENDENCIES monitors
              OPENCV_DEPENDENCIES highgui)
