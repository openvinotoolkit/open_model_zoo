# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_package(OpenCV REQUIRED COMPONENTS core imgproc)

set(SOURCES presenter.cpp cpu_monitor.cpp memory_monitor.cpp)
set(HEADERS presenter.h cpu_monitor.h memory_monitor.h)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${SOURCES})
source_group("include" FILES ${HEADERS})

add_library(monitors STATIC ${SOURCES} ${HEADERS})
target_include_directories(monitors PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(monitors PRIVATE opencv_core opencv_imgproc)
if(WIN32)
    target_link_libraries(monitors PRIVATE pdh) # for CPU, GPU utilization
    target_link_libraries(monitors PRIVATE Psapi) # for memory monitor
endif()
