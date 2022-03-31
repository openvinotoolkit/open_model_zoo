/*
// Copyright (C) 2020-2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "utils/config_factory.h"

#include <set>
#include <string>
#include <utility>
#include <vector>

#include <openvino/runtime/intel_gpu/properties.hpp>

#include "utils/args_helper.hpp"

std::set<std::string> ModelConfig::getDevices() {
    if (devices.empty()) {
        for (const std::string& device : parseDevices(deviceName)) {
            devices.insert(device);
        }
    }

    return devices;
}

ModelConfig ConfigFactory::getUserConfig(const std::string& flags_d,
                                         uint32_t flags_nireq,
                                         const std::string& flags_nstreams,
                                         uint32_t flags_nthreads) {
    auto config = getCommonConfig(flags_d, flags_nireq);

    std::map<std::string, int> deviceNstreams = parseValuePerDevice(config.getDevices(), flags_nstreams);
    for (const auto& device : config.getDevices()) {
        if (device == "CPU") {  // CPU supports a few special performance-oriented keys
            // limit threading for CPU portion of inference
            if (flags_nthreads != 0)
                config.compiledModelConfig.emplace(ov::inference_num_threads.name(), flags_nthreads);

            config.compiledModelConfig.emplace(ov::affinity.name(), ov::Affinity::NONE);

            ov::streams::Num nstreams =
                deviceNstreams.count(device) > 0 ? ov::streams::Num(deviceNstreams[device]) : ov::streams::AUTO;
            config.compiledModelConfig.emplace(ov::streams::num.name(), nstreams);
        } else if (device == "GPU") {
            ov::streams::Num nstreams =
                deviceNstreams.count(device) > 0 ? ov::streams::Num(deviceNstreams[device]) : ov::streams::AUTO;
            config.compiledModelConfig.emplace(ov::streams::num.name(), nstreams);
            if (flags_d.find("MULTI") != std::string::npos &&
                config.getDevices().find("CPU") != config.getDevices().end()) {
                // multi-device execution with the CPU + GPU performs best with GPU throttling hint,
                // which releases another CPU thread (that is otherwise used by the GPU driver for active polling)
                config.compiledModelConfig.emplace(ov::intel_gpu::hint::queue_throttle.name(),
                                                   ov::intel_gpu::hint::ThrottleLevel(1));
            }
        }
    }
    return config;
}

ModelConfig ConfigFactory::getMinLatencyConfig(const std::string& flags_d, uint32_t flags_nireq) {
    auto config = getCommonConfig(flags_d, flags_nireq);
    for (const auto& device : config.getDevices()) {
        if (device == "CPU") {  // CPU supports a few special performance-oriented keys
            config.compiledModelConfig.emplace(ov::streams::num.name(), 1);
        } else if (device == "GPU") {
            config.compiledModelConfig.emplace(ov::streams::num.name(), 1);
        }
    }
    return config;
}

ModelConfig ConfigFactory::getCommonConfig(const std::string& flags_d, uint32_t flags_nireq) {
    ModelConfig config;

    if (!flags_d.empty()) {
        config.deviceName = flags_d;
    }

    config.maxAsyncRequests = flags_nireq;

    return config;
}

std::map<std::string, std::string> ModelConfig::getLegacyConfig() {
    std::map<std::string, std::string> config;
    for (const auto& item : compiledModelConfig) {
        config[item.first] = item.second.as<std::string>();
    }
    return config;
}
