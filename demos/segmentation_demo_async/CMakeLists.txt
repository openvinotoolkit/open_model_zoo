# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

FILE(GLOB SRC_FILES ./*.cpp)
FILE(GLOB H_FILES ./*.h)

ie_add_sample(NAME segmentation_demo_async
              SOURCES ${SRC_FILES}
              HEADERS ${H_FILES}
              DEPENDENCIES monitors models_den
              OPENCV_DEPENDENCIES highgui videoio imgproc)

target_link_libraries(segmentation_demo_async PRIVATE ngraph::ngraph)
