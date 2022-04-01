// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <details/ie_exception.hpp>

#define PT_CHECK(cond) IE_ASSERT(cond) << " "

#define PT_CHECK_BINARY(actual, expected, op) \
    IE_ASSERT(actual op expected) << ". " << actual << " vs " << expected << ".  "

#define PT_CHECK_EQ(actual, expected) PT_CHECK_BINARY(actual, expected, ==)
#define PT_CHECK_NE(actual, expected) PT_CHECK_BINARY(actual, expected, !=)
#define PT_CHECK_LT(actual, expected) PT_CHECK_BINARY(actual, expected, <)
#define PT_CHECK_GT(actual, expected) PT_CHECK_BINARY(actual, expected, >)
#define PT_CHECK_LE(actual, expected) PT_CHECK_BINARY(actual, expected, <=)
#define PT_CHECK_GE(actual, expected) PT_CHECK_BINARY(actual, expected, >=)
