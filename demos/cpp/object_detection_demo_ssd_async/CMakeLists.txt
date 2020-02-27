# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME object_detection_demo_ssd_async
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/object_detection_demo_ssd_async.hpp"
              DEPENDENCIES monitors
              OPENCV_DEPENDENCIES highgui videoio imgproc)

target_link_libraries(object_detection_demo_ssd_async PRIVATE ngraph::ngraph)
