# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB SOURCES ./src/*.cpp)
file(GLOB HEADERS ./include/pipelines/*.h)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${SOURCES})
source_group("include" FILES ${HEADERS})

add_library(pipelines STATIC ${SOURCES} ${HEADERS})
target_include_directories(pipelines PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(pipelines PRIVATE gflags ${InferenceEngine_LIBRARIES} models utils opencv_core opencv_imgproc)
