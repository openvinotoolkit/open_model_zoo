# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_VA)

file(GLOB SRC_FILES ./*.cpp)
file(GLOB H_FILES ./*.h)

add_demo(NAME va_object_detection_demo
    SOURCES ${SRC_FILES}
    HEADERS ${H_FILES}
    DEPENDENCIES monitors models pipelines va_video)

endif()