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

add_library(monitors_extension MODULE monitors_extension.cpp)
target_include_directories(monitors_extension PRIVATE ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(monitors_extension PRIVATE ${PYTHON_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(monitors_extension PRIVATE monitors)
target_include_directories(monitors_extension PRIVATE "${PROJECT_SOURCE_DIR}/common" {NUMPY_INCLUDE_DIR})
set_target_properties(monitors_extension PROPERTIES PREFIX "" OUTPUT_NAME "monitors_extension")
if(WIN32)
    set_target_properties(monitors_extension PROPERTIES SUFFIX ".pyd")
endif()
