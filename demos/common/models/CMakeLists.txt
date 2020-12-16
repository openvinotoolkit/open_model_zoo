# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(ngraph REQUIRED)

FILE(GLOB SOURCES ./src/*.cpp)
FILE(GLOB HEADERS ./include/models/*.h)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${SOURCES})
source_group("include" FILES ${HEADERS})

add_library(models STATIC ${SOURCES} ${HEADERS})
target_include_directories(models PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(models PRIVATE ngraph::ngraph gflags ${InferenceEngine_LIBRARIES} common opencv_core opencv_imgproc)
