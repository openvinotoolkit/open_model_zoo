# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_package (PkgConfig)

pkg_check_modules (GSTREAMER REQUIRED gstreamer-1.0 gstreamer-base-1.0 gstreamer-app-1.0 gstreamer-allocators-1.0 gstreamer-video-1.0)
pkg_search_module (VA va libva REQUIRED)
pkg_search_module (VADRM va-drm libva-drm REQUIRED)

FILE(GLOB SOURCES ./src/*.cpp)
FILE(GLOB HEADERS ./include/*.h)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${SOURCES})
source_group("include" FILES ${HEADERS})

add_library(va_video STATIC ${SOURCES} ${HEADERS})
target_include_directories(va_video PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ${VA_INCLUDE_DIRS}
        ${GSTREAMER_INCLUDE_DIRS}
        ${VA_INCLUDE_DIRS}
        ${VADRM_INCLUDE_DIRS}
)
target_link_libraries(va_video PRIVATE ${VA_LIBRARIES}
        ${GSTREAMER_LIBRARIES}
        ${VA_LIBRARIES}
        ${VADRM_LIBRARIES}
        /usr/lib/x86_64-linux-gnu/libOpenCL.so
)
