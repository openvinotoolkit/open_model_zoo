// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with helper functions for InferenceEngine configuration
 * @file ie_config_helper.hpp
 */

#pragma once

#include <cldnn/cldnn_config.hpp>

#include "samples/common.hpp"
#include "samples/args_helper.hpp"

using namespace InferenceEngine;

void configureInferenceEngine(Core& ie,
                              std::string& deviceString,
                              std::string& deviceInfo,
                              const std::string& lString,
                              const std::string& cString,
                              bool pc);

std::map<std::string, std::string> createConfig(const std::string& deviceString,
                                                const std::string& nstreamsString,
                                                int nthreads,
                                                bool minLatency = false);
