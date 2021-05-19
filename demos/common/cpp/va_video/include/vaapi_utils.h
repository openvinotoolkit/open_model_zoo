/*******************************************************************************
 * Copyright (C) 2018-2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#pragma once

#include <stdexcept>
#include <stdio.h>
#include <string>
#include <va/va.h>

#define VA_CALL(_FUNC)                                                                                                 \
{                                                                                                                  \
    VAStatus _status = _FUNC;                                                                                      \
    if (_status != VA_STATUS_SUCCESS) {                                                                            \
        throw std::runtime_error(#_FUNC " failed, sts=" + std::to_string(_status) + ": " + vaErrorStr(_status));   \
    } /*else {                                                                                                     \
        printf(#_FUNC " returned %d\n", _status);                                                                  \
    }*/                                                                                                            \
}
