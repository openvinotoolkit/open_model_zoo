# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc)

set(TARGET_NAME pose_extractor)
add_library(${TARGET_NAME} MODULE  ${CMAKE_CURRENT_SOURCE_DIR}/wrapper.cpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/src/extract_poses.hpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/src/extract_poses.cpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/src/human_pose.hpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/src/human_pose.cpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/src/peak.hpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/src/peak.cpp)

check_code_style(NAME ${TARGET_NAME}
    CLANG_FORMAT ${ENABLE_CLANG_FORMAT}
    CPPLINT ${ENABLE_CPPLINT})

target_include_directories(${TARGET_NAME} PRIVATE src/ ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
target_link_libraries(${TARGET_NAME} ${PYTHON_LIBRARIES} opencv_core opencv_imgproc)
set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "")
if(WIN32)
    set_target_properties(${TARGET_NAME} PROPERTIES SUFFIX ".pyd")
endif()
