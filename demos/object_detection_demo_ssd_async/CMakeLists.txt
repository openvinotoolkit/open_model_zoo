# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
file (GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file (GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
# message(WARNING ${SOURCES})


ie_add_sample(NAME object_detection_demo_ssd_async
              SOURCES "${SOURCES}"
              HEADERS "${HEADERS}"
              DEPENDENCIES monitors
              OPENCV_DEPENDENCIES core highgui videoio)

target_link_libraries(object_detection_demo_ssd_async PRIVATE ngraph::ngraph)
