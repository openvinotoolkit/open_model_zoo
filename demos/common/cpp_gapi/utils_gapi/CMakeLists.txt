# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*")
file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*")

set(TARGET_NAME "utils_gapi")
source_group("src" FILES ${SOURCES})
source_group("include" FILES ${HEADERS})

add_library(${TARGET_NAME} STATIC ${HEADERS} ${SOURCES})

check_code_style(NAME ${TARGET_NAME}
    CLANG_FORMAT ${ENABLE_CLANG_FORMAT}
    CPPLINT ${ENABLE_CPPLINT})

target_include_directories(utils_gapi PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(utils_gapi PRIVATE gflags openvino::runtime opencv_core utils)
