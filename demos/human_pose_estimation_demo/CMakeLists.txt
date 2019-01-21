# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "human_pose_estimation_demo")

# Find OpenCV components if exist
find_package(OpenCV COMPONENTS highgui QUIET)
if(NOT(OpenCV_FOUND))
    message(WARNING "OPENCV is disabled or not found, " ${TARGET_NAME} " skipped")
    return()
endif()

set(SOURCES include/human_pose.hpp
            include/human_pose_estimation_demo.hpp
            include/human_pose_estimator.hpp
            include/peak.hpp
            include/render_human_pose.hpp
            src/human_pose.cpp
            src/human_pose_estimator.cpp
            src/peak.cpp
            src/render_human_pose.cpp
            main.cpp)

add_executable(${TARGET_NAME} ${SOURCES})

link_directories(${LIB_FOLDER})

set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE"
  COMPILE_PDB_NAME ${TARGET_NAME})

target_include_directories(${TARGET_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} ${OpenCV_LIBS} gflags)

if(UNIX)
  target_link_libraries(${TARGET_NAME} ${LIB_DL} pthread)
endif()
