# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "pedestrian_tracker_demo")

# Find OpenCV components if exist
find_package(OpenCV COMPONENTS highgui QUIET)
if(NOT(OpenCV_FOUND))
    message(WARNING "OPENCV is disabled or not found, " ${TARGET_NAME} " skipped")
    return()
endif()

set(sources
    include/cnn.hpp
    include/core.hpp
    include/descriptor.hpp
    include/detector.hpp
    include/distance.hpp
    include/image_reader.hpp
    include/kuhn_munkres.hpp
    include/logging.hpp
    include/pedestrian_tracker_demo.hpp
    include/tracker.hpp
    include/utils.hpp
    src/cnn.cpp
    src/core.cpp
    src/detector.cpp
    src/distance.cpp
    src/image_reader.cpp
    src/kuhn_munkres.cpp
    src/tracker.cpp
    src/utils.cpp
    main.cpp)

add_executable(${TARGET_NAME} ${sources})
link_directories(${LIB_FOLDER})
add_dependencies(${TARGET_NAME} gflags)
set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE"
  COMPILE_PDB_NAME ${TARGET_NAME})

target_include_directories(${TARGET_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_link_libraries(${TARGET_NAME} IE::ie_cpu_extension ${InferenceEngine_LIBRARIES} ${OpenCV_LIBS} gflags)
if(UNIX)
  target_link_libraries(${TARGET_NAME} ${LIB_DL} pthread)
endif()

