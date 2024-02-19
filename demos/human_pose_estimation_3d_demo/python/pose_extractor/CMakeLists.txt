# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc)

set(target_name pose_extractor)
add_library(${target_name} MODULE wrapper.cpp
                                  src/extract_poses.hpp src/extract_poses.cpp
                                  src/human_pose.hpp src/human_pose.cpp
                                  src/peak.hpp src/peak.cpp)
target_include_directories(${target_name} PRIVATE src/ ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
target_link_libraries(${target_name} ${PYTHON_LIBRARIES} opencv_core opencv_imgproc)
set_target_properties(${target_name} PROPERTIES PREFIX "")
if(WIN32)
    set_target_properties(${target_name} PROPERTIES SUFFIX ".pyd")
endif()
