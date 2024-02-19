# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB SRC_FILES ./*.cpp)
file(GLOB H_FILES ./*.h)

add_demo(NAME object_detection_demo
    SOURCES ${SRC_FILES}
    HEADERS ${H_FILES}
    DEPENDENCIES monitors models pipelines)
