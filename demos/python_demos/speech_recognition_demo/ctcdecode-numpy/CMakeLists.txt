# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

execute_process(
COMMAND "${PYTHON_EXECUTABLE}" -m pip install .
WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python_demos/speech_recognition_demo/ctcdecode-numpy
RESULT_VARIABLE CTCDECODE_NUMPY_INTALL_FAILED)
if(CTCDECODE_NUMPY_INTALL_FAILED)
    message(FATAL_ERROR "Failed to pip-install ctcdecode_numpy package")
endif()
