# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_monitor.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/memory_monitor.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/presenter.cpp")

set(HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/monitors/cpu_monitor.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/monitors/memory_monitor.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/monitors/presenter.h")

if(WIN32)
    list(APPEND SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/query_wrapper.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/query_wrapper.h")
endif()

set(TARGET_NAME "monitors")
# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${SOURCES})
source_group("include" FILES ${HEADERS})

add_library(${TARGET_NAME} STATIC ${SOURCES} ${HEADERS})

check_code_style(NAME ${TARGET_NAME}
    CLANG_FORMAT ${ENABLE_CLANG_FORMAT}
    CPPLINT ${ENABLE_CPPLINT})

target_include_directories(monitors PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(monitors PRIVATE opencv_core opencv_imgproc)
if(WIN32)
    target_link_libraries(monitors PRIVATE pdh)
endif()
