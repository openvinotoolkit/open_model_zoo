# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

project(multicam)

find_package(Threads REQUIRED)
find_package(Boost REQUIRED)

set(SOURCES
    controller.cpp
    camera.cpp
    utils.cpp)

set(HEADERS
    controller.hpp
    camera.hpp
    utils.hpp)

add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} PUBLIC
    Threads::Threads)
target_include_directories(${PROJECT_NAME} PUBLIC ${Boost_INCLUDE_DIR})
