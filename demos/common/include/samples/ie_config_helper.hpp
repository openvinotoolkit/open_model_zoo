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

#include <cldnn/cldnn_config.hpp>
#include <vpu/myriad_config.hpp>

std::string formatDeviceString(const std::string& deviceString);

std::map<std::string, std::string> createConfig(const std::string& deviceString,
                                                const std::string& nstreamsString,
                                                int nthreads,
                                                bool minLatency = false);

std::map<std::string, std::string> createDefaultConfig(std::string deviceString);
