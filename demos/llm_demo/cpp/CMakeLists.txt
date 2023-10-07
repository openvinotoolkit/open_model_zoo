# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_demo(NAME llm_demo SOURCES main.cpp DEPENDENCIES llama common)
if(MSVC)  # Disable the warning from llama.cpp
    target_compile_options(llm_demo PRIVATE /wd4996)
endif()
