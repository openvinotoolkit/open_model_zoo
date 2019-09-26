# Copyright (C) 2018-2019 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(TARGET_NAME "multi-channel-face-detection-demo")

if( BUILD_DEMO_NAME AND NOT ${BUILD_DEMO_NAME} STREQUAL ${TARGET_NAME} )
    message(STATUS "DEMO ${TARGET_NAME} SKIPPED")
    return()
endif()

# Find OpenCV components if exist
find_package(OpenCV COMPONENTS highgui QUIET)
if(NOT(OpenCV_FOUND))
    message(WARNING "OPENCV is disabled or not found, " ${TARGET_NAME} " skipped")
    return()
endif()

file (GLOB MAIN_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

file (GLOB MAIN_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
        )

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${MAIN_SRC})
source_group("include" FILES ${MAIN_HEADERS})

# Create library file from sources.
add_executable(${TARGET_NAME} ${MAIN_SRC} ${MAIN_HEADERS})

set_target_properties(${TARGET_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    COMPILE_PDB_NAME ${TARGET_NAME})

if(MULTICHANNEL_DEMO_USE_TBB)
    find_package(TBB REQUIRED tbb)
    target_link_libraries(${TARGET_NAME} ${TBB_IMPORTED_TARGETS})
    target_compile_definitions(${TARGET_NAME} PRIVATE
        USE_TBB=1
        __TBB_ALLOW_MUTABLE_FUNCTORS=1)

    if(FALSE) # disable task isolation for now due to bugs in tbb
        target_compile_definitions(${TARGET_NAME} PRIVATE
            TBB_PREVIEW_TASK_ISOLATION=1
            TBB_TASK_ISOLATION=1)
    endif()
endif()

if(TARGET IE::ie_cpu_extension)
    add_definitions(-DWITH_EXTENSIONS)
endif()

target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} gflags ${OpenCV_LIBRARIES} common)

if(TARGET IE::ie_cpu_extension)
    target_link_libraries(${TARGET_NAME} IE::ie_cpu_extension)
endif()

if(UNIX)
    target_link_libraries( ${TARGET_NAME} pthread)
endif()

if(COMMAND add_cpplint_target)
    add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})
endif()

if(NOT TARGET ie_samples)
    add_custom_target(ie_samples ALL)
endif()

add_dependencies(ie_samples ${TARGET_NAME})
