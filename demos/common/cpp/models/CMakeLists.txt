# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB SOURCES ./src/*.cpp)
file(GLOB HEADERS ./include/models/*.h)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${SOURCES})
source_group("include" FILES ${HEADERS})

add_library(models STATIC ${SOURCES} ${HEADERS})
target_include_directories(models PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(models PRIVATE gflags openvino::runtime utils opencv_core opencv_imgproc)
