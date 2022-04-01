# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ctcdecode_numpy_impl)

file(GLOB_RECURSE HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

source_group("include" FILES ${HEADERS})
source_group("src" FILES ${SOURCES})

add_library(${TARGET_NAME} MODULE ${HEADERS} ${SOURCES})

# check_code_style(NAME ${TARGET_NAME}
#     CLANG_FORMAT ${ENABLE_CLANG_FORMAT}
#     CPPLINT ${ENABLE_CPPLINT})

target_include_directories(${TARGET_NAME} PRIVATE
    ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR} third_party/ThreadPool)
target_link_libraries(${TARGET_NAME} ${PYTHON_LIBRARIES})

set_target_properties(${TARGET_NAME} PROPERTIES
    PREFIX ""
    OUTPUT_NAME _impl
)

if(WIN32)
    set_target_properties(${TARGET_NAME} PROPERTIES SUFFIX ".pyd")
endif()

foreach(artifact IN ITEMS ARCHIVE LIBRARY PDB RUNTIME)
    set_property(TARGET ${TARGET_NAME} APPEND_STRING
        PROPERTY "${artifact}_OUTPUT_DIRECTORY" "/ctcdecode_numpy")
endforeach()

# SWIG-generated code causes some warnings; disable them.
if(COMPILER_IS_GCC_LIKE)
    target_compile_options(${TARGET_NAME} PRIVATE -Wno-narrowing)
elseif(MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE
        /wd4244  # Disable conversion warning =
        /wd4838  # Disable conversion warning in aggregate or list initialization
    )
endif()

# The Python files have to be in the same directory as the native module,
# because a non-namespace package (i.e., a package with an __init__.py file)
# has to have all of its modules in the same directory.
# So copy the Python files over.
# It would be preferable to use add_custom_command so that we don't try
# to copy files that are already up-to-date, but the output path depends
# on the current configuration, and add_custom_command doesn't support that.
add_custom_target(ctcdecode_numpy ALL
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/ctcdecode_numpy/__init__.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/ctcdecode_numpy/impl.py"
        "$<TARGET_FILE_DIR:${TARGET_NAME}>"
    VERBATIM
)

add_dependencies(ctcdecode_numpy ${TARGET_NAME})
