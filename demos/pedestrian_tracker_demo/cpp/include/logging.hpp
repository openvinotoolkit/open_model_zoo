// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <opencv2/core.hpp>

#define PT_CHECK(cond) CV_Assert(cond);

#define PT_CHECK_BINARY(actual, expected, op) \
    CV_Assert(actual op expected);

#define PT_CHECK_EQ(actual, expected) PT_CHECK_BINARY(actual, expected, ==)
#define PT_CHECK_NE(actual, expected) PT_CHECK_BINARY(actual, expected, !=)
#define PT_CHECK_LT(actual, expected) PT_CHECK_BINARY(actual, expected, <)
#define PT_CHECK_GT(actual, expected) PT_CHECK_BINARY(actual, expected, >)
#define PT_CHECK_LE(actual, expected) PT_CHECK_BINARY(actual, expected, <=)
#define PT_CHECK_GE(actual, expected) PT_CHECK_BINARY(actual, expected, >=)
