# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NUMPY_NOT_FOUND)
if(NUMPY_NOT_FOUND)
    message(FATAL_ERROR "NumPy headers not found")
endif()

find_package(OpenCV 4 REQUIRED)

set(target_name pose_extractor)
add_library(${target_name} MODULE wrapper.cpp
                                  src/extract_poses.hpp src/extract_poses.cpp
                                  src/human_pose.hpp src/human_pose.cpp
                                  src/peak.hpp src/peak.cpp)
target_include_directories(${target_name} PRIVATE src/ ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${target_name} ${PYTHON_LIBRARIES} ${OpenCV_LIBS})
set_target_properties(${target_name} PROPERTIES PREFIX "" OUTPUT_NAME "${target_name}")
if(WIN32)
    set_target_properties(${target_name} PROPERTIES SUFFIX ".pyd")
endif()

