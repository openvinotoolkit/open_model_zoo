#===============================================================================
# Copyright 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

set(TARGET_NAME "cpu_extension")

file(GLOB_RECURSE SRC *.cpp)
file(GLOB_RECURSE HDR *.hpp)

if(WIN32)
    add_definitions(-DIMPLEMENT_INFERENCE_ENGINE_API)
endif()

include_directories (PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${InferenceEngine_INCLUDE_DIRS}
)

enable_omp()

add_library(${TARGET_NAME} SHARED ${SRC} ${HDR})
target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} ${intel_omp_lib})
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})

set_target_cpu_flags(${TARGET_NAME})
