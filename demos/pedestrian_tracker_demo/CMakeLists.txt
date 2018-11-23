# Copyright (c) 2018 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set (TARGET_NAME "pedestrian_tracker_demo")

if( BUILD_DEMO_NAME AND NOT ${BUILD_DEMO_NAME} STREQUAL ${TARGET_NAME} )
    message(STATUS "DEMO ${TARGET_NAME} SKIPPED")
    return()
endif()

find_package(Threads)
find_package(OpenCV)
if(NOT(OpenCV_FOUND))
    message(STATUS "OPENCV is disabled or not found, " ${TARGET_NAME} " skipped")
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
target_include_directories(${TARGET_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} IE::ie_cpu_extension ${InferenceEngine_LIBRARIES} ${OpenCV_LIBS} gflags)
if(UNIX)
  target_link_libraries(${TARGET_NAME} ${LIB_DL} pthread)
endif()

