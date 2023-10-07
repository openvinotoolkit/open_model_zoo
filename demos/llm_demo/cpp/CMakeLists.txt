# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(COMPILER_IS_GCC_LIKE)  # Disable the warnings from llama.cpp
    add_compile_options(-Wno-implicit-fallthrough -Wno-unused-function)
elseif(MSVC)
    add_compile_options(/wd4146 /wd4305 /wd4715 /wd4996)
endif()
add_demo(NAME llm_demo SOURCES main.cpp DEPENDENCIES llama common)
