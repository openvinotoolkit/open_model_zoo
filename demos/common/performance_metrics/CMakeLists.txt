# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_package(InferenceEngine 2.0 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS imgproc)

set(SOURCES metrics.cpp)
set(HEADERS metrics.h)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${SOURCES})
source_group("include" FILES ${HEADERS})

add_library(performance_metrics STATIC ${SOURCES} ${HEADERS} ${InferenceEngine_INCLUDE_DIRS})
target_include_directories(performance_metrics PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(performance_metrics ${InferenceEngine_LIBRARIES} opencv_core opencv_imgproc)
