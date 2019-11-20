// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pdh.h>
class QueryWrapper {
public:
    QueryWrapper();
    ~QueryWrapper();
    operator PDH_HQUERY() const;
    PDH_HQUERY query;
};
