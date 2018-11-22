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

cmake_minimum_required(VERSION 2.8)

set(TARGET_NAME "multi-channel-demo")

if( BUILD_DEMO_NAME AND NOT ${BUILD_DEMO_NAME} STREQUAL ${TARGET_NAME} )
    message(STATUS "DEMO ${TARGET_NAME} SKIPPED")
    return()
endif()

find_package(OpenCV)
if(NOT(OpenCV_FOUND))
    message(STATUS "OPENCV is disabled or not found, " ${TARGET_NAME} " skipped")
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

link_directories(${LIB_FOLDER})

# Create library file from sources.
add_executable(${TARGET_NAME} ${MAIN_SRC} ${MAIN_HEADERS})

target_include_directories(${TARGET_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

add_dependencies(${TARGET_NAME} gflags)

set_target_properties(${TARGET_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    COMPILE_PDB_NAME ${TARGET_NAME})

option(MULTICHANNEL_FD_DEMO_USE_TBB "Use TBB-based threading in multichannel face detection demo" OFF)
option(MULTICHANNEL_FD_DEMO_USE_NATIVE_CAM "Use native camera api in multichannel face detection demo" OFF)

if(MULTICHANNEL_FD_DEMO_USE_TBB)
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

if(MULTICHANNEL_FD_DEMO_USE_NATIVE_CAM)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    add_subdirectory(multicam)
    target_link_libraries(${TARGET_NAME} multicam)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        USE_NATIVE_CAMERA_API=1)

    # LIBVA_INCLUDE_DIR
    # LIBVA_LIB_DIR
    if ((NOT LIBVA_INCLUDE_DIR) OR (NOT LIBVA_LIB_DIR))
        message(FATAL_ERROR "\"LIBVA_INCLUDE_DIR\" and \"LIBVA_LIB_DIR\" must be set")
    endif()
    unset(_LIBVA_LIB CACHE)
    find_library(_LIBVA_LIB NAMES
        libva.so
        libva.so.2
        NO_DEFAULT_PATH
        PATHS ${LIBVA_LIB_DIR})
    unset(_LIBVA_X11_LIB CACHE)
    find_library(_LIBVA_X11_LIB NAMES
        libva-x11.so
        libva-x11.so.2
        NO_DEFAULT_PATH
        PATHS ${LIBVA_LIB_DIR})
    unset(_LIBVA_DRM_LIB CACHE)
    find_library(_LIBVA_DRM_LIB NAMES
        libva-drm.so
        libva-drm.so.2
        NO_DEFAULT_PATH
        PATHS ${LIBVA_LIB_DIR})
    unset(_LIBVA_INCLUDE_DIR CACHE)
    find_path(_LIBVA_INCLUDE_DIR
      NAMES "va/va.h"
      NO_DEFAULT_PATH
      PATHS ${LIBVA_INCLUDE_DIR}
    )
    if ((NOT _LIBVA_LIB) OR
        (NOT _LIBVA_X11_LIB) OR
        (NOT _LIBVA_DRM_LIB) OR
        (NOT _LIBVA_INCLUDE_DIR))
        message(FATAL_ERROR "libva not found")
    endif()
    message(STATUS "libva:")
    message(STATUS "    \"${_LIBVA_LIB}\"")
    message(STATUS "    \"${_LIBVA_X11_LIB}\"")
    message(STATUS "    \"${_LIBVA_DRM_LIB}\"")
    message(STATUS "    \"${_LIBVA_INCLUDE_DIR}\"")

    target_include_directories(${TARGET_NAME} PRIVATE ${_LIBVA_INCLUDE_DIR})
    target_link_libraries(${TARGET_NAME}
        X11
        ${_LIBVA_LIB}
        ${_LIBVA_X11_LIB}
        ${_LIBVA_DRM_LIB}
        )
    target_compile_definitions(${TARGET_NAME} PRIVATE
        USE_LIBVA=1)
endif()

target_link_libraries(${TARGET_NAME} IE::ie_cpu_extension ${InferenceEngine_LIBRARIES} gflags ${OpenCV_LIBRARIES})

if(UNIX)
    target_link_libraries( ${TARGET_NAME} ${LIB_DL} pthread)
endif()
