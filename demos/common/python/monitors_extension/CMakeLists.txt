# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

find_package(OpenCV 4 REQUIRED COMPONENTS core)

add_library(monitors_extension MODULE monitors_extension.cpp)
target_include_directories(monitors_extension PRIVATE ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
target_link_libraries(monitors_extension PRIVATE ${PYTHON_LIBRARIES} opencv_core monitors utils)
set_target_properties(monitors_extension PROPERTIES PREFIX "")
if(WIN32)
    set_target_properties(monitors_extension PROPERTIES SUFFIX ".pyd")
endif()
