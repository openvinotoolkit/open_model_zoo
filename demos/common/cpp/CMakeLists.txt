# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_VA)
    add_definitions(-DUSE_VA)
    add_subdirectory(va_video)
endif()

add_subdirectory(utils)
add_subdirectory(monitors)
add_subdirectory(models)
add_subdirectory(pipelines)
