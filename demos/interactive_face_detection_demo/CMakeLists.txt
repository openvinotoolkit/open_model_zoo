# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file (GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB MAIN_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h*)

ie_add_sample(NAME interactive_face_detection_demo
              SOURCES ${MAIN_SRC}
              HEADERS ${MAIN_HEADERS}
              DEPENDENCIES monitors
              OPENCV_DEPENDENCIES highgui)

target_link_libraries(interactive_face_detection_demo PRIVATE ngraph::ngraph)
