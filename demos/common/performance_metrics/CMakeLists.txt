# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(SOURCES metrics.cpp)
set(HEADERS metrics.h)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${SOURCES})
source_group("include" FILES ${HEADERS})

add_library(performance_metrics STATIC ${SOURCES} ${HEADERS})
target_include_directories(performance_metrics PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
