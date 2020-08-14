// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief ?????
 * @file ie_config_helper.hpp
 */

#pragma once

#include <cldnn/cldnn_config.hpp>
#include <ie_plugin_config.hpp>

#include "samples/args_helper.hpp"

/**
* @brief ?????
* @param arg ?????
* @return ?????
*/
std::map<std::string, std::string> createConfig(const std::string& deviceString,
                                                const std::string& nstreamsString,
                                                const int& nthreads,
                                                bool minLatency = false);
