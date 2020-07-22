# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs videoio)

set(SOURCES images_capture.cpp)
set(HEADERS images_capture.h)
# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${SOURCES})
source_group("include" FILES ${HEADERS})

add_library(images_capture STATIC ${SOURCES} ${HEADERS})
target_include_directories(images_capture PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/.." PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(images_capture PRIVATE opencv_core opencv_imgcodecs opencv_videoio)
