# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "monitors_extension")

find_package(OpenCV 4 REQUIRED COMPONENTS core)

add_library(${TARGET_NAME} MODULE  ${CMAKE_CURRENT_SOURCE_DIR}/monitors_extension.cpp)

check_code_style(NAME ${TARGET_NAME}
    CLANG_FORMAT ${ENABLE_CLANG_FORMAT}
    CPPLINT ${ENABLE_CPPLINT})

target_include_directories(${TARGET_NAME} PRIVATE ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
target_link_libraries(${TARGET_NAME} PRIVATE ${PYTHON_LIBRARIES} opencv_core monitors utils)
set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "")
if(WIN32)
    set_target_properties(${TARGET_NAME} PROPERTIES SUFFIX ".pyd")
endif()
