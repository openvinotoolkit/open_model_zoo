# Copyright (C) 2022 Wahtari GmbH
# SPDX-License-Identifier: Apache-2.0
#

set(SOURCES
    src/Camera.cpp)

set(HEADERS
    include/ncam/BufferedChannel.hpp    
    include/ncam/Camera.hpp
    include/ncam/MJPEGStreamer.hpp)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${SOURCES})
source_group("include" FILES ${HEADERS})

# AlliedVision Vimba SDK.
set(Vimba_LIBS VimbaC VimbaCPP)
include_directories("/usr/local/include/VimbaCPP/Include")

# Wahtari Controller-libs.
set(nLab_Ctrl_LIBS nlab-ctrl nlab-ctrl-cpp)
include_directories("/usr/local/include/controller-libs")

add_library(ncam STATIC ${SOURCES} ${HEADERS})
target_include_directories(ncam PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(ncam PRIVATE opencv_core opencv_imgproc ${Vimba_LIBS} ${nLab_Ctrl_LIBS})
