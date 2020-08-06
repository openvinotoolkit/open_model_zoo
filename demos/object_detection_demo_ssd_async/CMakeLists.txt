# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

FILE(GLOB SRC_FILES ./*.cpp ../common/processing/*.cpp)
FILE(GLOB H_FILES ./*.h ../common/processing/*.h)

ie_add_sample(NAME object_detection_demo_ssd_async
              SOURCES ${SRC_FILES}
              HEADERS ${H_FILES}
              DEPENDENCIES monitors
              OPENCV_DEPENDENCIES highgui videoio imgproc)

target_include_directories(object_detection_demo_ssd_async PRIVATE ../common/processing/)
target_link_libraries(object_detection_demo_ssd_async PRIVATE ngraph::ngraph)
