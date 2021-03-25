# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(SOURCES
    src/cpu_monitor.cpp
    src/memory_monitor.cpp
    src/presenter.cpp)

set(HEADERS
    include/monitors/cpu_monitor.h
    include/monitors/memory_monitor.h
    include/monitors/presenter.h)

if(WIN32)
    list(APPEND SOURCES src/query_wrapper.cpp src/query_wrapper.h)
endif()
# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${SOURCES})
source_group("include" FILES ${HEADERS})

add_library(monitors STATIC ${SOURCES} ${HEADERS})
target_include_directories(monitors PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(monitors PRIVATE opencv_core opencv_imgproc)
if(WIN32)
    target_link_libraries(monitors PRIVATE pdh)
endif()
