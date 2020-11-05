# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_package(OpenCV REQUIRED COMPONENTS core imgproc)

FILE(GLOB SOURCES ./src/*.cpp)
FILE(GLOB HEADERS ./include/*.h)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${SOURCES})
source_group("include" FILES ${HEADERS})

add_library(models_den STATIC ${SOURCES} ${HEADERS})
target_include_directories(models_den PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_link_libraries(models_den PRIVATE gflags ${InferenceEngine_LIBRARIES} opencv_core opencv_imgproc)
if(WIN32)
    target_link_libraries(models_den PRIVATE pdh)
endif()
