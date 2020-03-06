# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

ie_add_sample(NAME object_detection_demo_faster_rcnn
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/object_detection_demo_faster_rcnn.h"
                      "${CMAKE_CURRENT_SOURCE_DIR}/detectionoutput.h"
              OPENCV_DEPENDENCIES core imgcodecs imgproc)
