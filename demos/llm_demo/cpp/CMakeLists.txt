# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_demo(NAME llm_demo SOURCES main.cpp DEPENDENCIES llama common)
if(COMPILER_IS_GCC_LIKE)  # Disable the warnings from llama.cpp
    target_compile_options(llm_demo BEFORE PUBLIC -Wno-implicit-fallthrough -Wno-unused-function)
elseif(MSVC)
    target_compile_options(llm_demo PRIVATE /wd4146 /wd4305 /wd4715 /wd4996)
endif()
