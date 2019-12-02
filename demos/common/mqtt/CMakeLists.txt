# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "mqtt")

file (GLOB MAIN_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

file (GLOB LIBRARY_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        )

find_path(MOSQUITTOPP_INCLUDE_DIR mosquitto.h)
find_library(MOSQUITTOPP_LIBRARY NAMES libmosquittopp mosquittopp)


# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})


# Create library file from sources.
add_library(${TARGET_NAME} SHARED ${MAIN_SRC} ${LIBRARY_HEADERS})

include_directories(${MOSQUITTOPP_INCLUDE_DIR})
target_link_libraries(${TARGET_NAME} ${MOSQUITTOPP_LIBRARY})

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_MQTT)

target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
                                                 "${CMAKE_CURRENT_SOURCE_DIR}/..")

set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})

# developer package

export(TARGETS ${TARGET_NAME} NAMESPACE IE::
       APPEND FILE "${CMAKE_BINARY_DIR}/targets_developer.cmake")
