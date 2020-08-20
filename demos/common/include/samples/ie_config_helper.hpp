// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with helper functions for InferenceEngine configuration
 * @file ie_config_helper.hpp
 */

#pragma once

#include <string>
#include <map>

void formatDeviceString(std::string& deviceString);

std::map<std::string, std::string> createSimpleConfig(const std::string& deviceString);

std::map<std::string, std::string> createConfig(const std::string& deviceString,
                                                const std::string& nstreamsString,
                                                int nthreads,
                                                bool minLatency = false);
