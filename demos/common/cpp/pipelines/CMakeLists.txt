# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB SOURCES ./src/*.cpp)
file(GLOB HEADERS ./include/pipelines/*.h)

set(TARGET_NAME "pipelines")
# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${SOURCES})
source_group("include" FILES ${HEADERS})

add_library(${TARGET_NAME} STATIC ${SOURCES} ${HEADERS})

check_code_style(${TARGET_NAME}
    ENABLE_CLANG_FORMAT
    ENABLE_CPPLINT)

target_include_directories(pipelines PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(pipelines PRIVATE gflags openvino::runtime models utils opencv_core opencv_imgproc)
