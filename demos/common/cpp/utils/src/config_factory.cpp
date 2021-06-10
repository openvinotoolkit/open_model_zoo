/*
// Copyright (C) 2018-2020 Intel Corporation
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

#include <utils/args_helper.hpp>
#include <utils/common.hpp>
#include <gpu/gpu_config.hpp>

using namespace InferenceEngine;

CnnConfig ConfigFactory::getUserConfig(const std::string& flags_d, const std::string& flags_l,
    const std::string& flags_c, bool flags_pc,
    uint32_t flags_nireq, const std::string& flags_nstreams, uint32_t flags_nthreads)
{
    auto config = getCommonConfig(flags_d, flags_l, flags_c, flags_pc, flags_nireq);
    std::set<std::string> devices;
    for (const std::string& device : parseDevices(flags_d)) {
        devices.insert(device);
    }
    std::map<std::string, unsigned> deviceNstreams = parseValuePerDevice(devices, flags_nstreams);
    for (const auto& device : devices) {
        if (device == "CPU") {  // CPU supports a few special performance-oriented keys
            // limit threading for CPU portion of inference
            if (flags_nthreads != 0)
                config.execNetworkConfig.emplace(CONFIG_KEY(CPU_THREADS_NUM), std::to_string(flags_nthreads));

            config.execNetworkConfig.emplace(CONFIG_KEY(CPU_BIND_THREAD), CONFIG_VALUE(NO));

            // for CPU execution, more throughput-oriented execution via streams
            config.execNetworkConfig.emplace(CONFIG_KEY(CPU_THROUGHPUT_STREAMS),
                (deviceNstreams.count(device) > 0 ? std::to_string(deviceNstreams.at(device))
                    : CONFIG_VALUE(CPU_THROUGHPUT_AUTO)));
        }
        else if (device == "GPU") {
            config.execNetworkConfig.emplace(CONFIG_KEY(GPU_THROUGHPUT_STREAMS),
                (deviceNstreams.count(device) > 0 ? std::to_string(deviceNstreams.at(device))
                    : CONFIG_VALUE(GPU_THROUGHPUT_AUTO)));

            if (flags_d.find("MULTI") != std::string::npos
                && devices.find("CPU") != devices.end()) {
                // multi-device execution with the CPU + GPU performs best with GPU throttling hint,
                // which releases another CPU thread (that is otherwise used by the GPU driver for active polling)
                config.execNetworkConfig.emplace(GPU_CONFIG_KEY(PLUGIN_THROTTLE), "1");
            }
        }
    }
    return config;
}

CnnConfig ConfigFactory::getMinLatencyConfig(const std::string& flags_d, const std::string& flags_l,
    const std::string& flags_c, bool flags_pc, uint32_t flags_nireq)
{
    auto config = getCommonConfig(flags_d, flags_l, flags_c, flags_pc, flags_nireq);
    std::set<std::string> devices;
    for (const std::string& device : parseDevices(flags_d)) {
        devices.insert(device);
    }
    for (const auto& device : devices) {
        if (device == "CPU") {  // CPU supports a few special performance-oriented keys
            config.execNetworkConfig.emplace(CONFIG_KEY(CPU_THROUGHPUT_STREAMS), "1");
        }
        else if (device == "GPU") {
            config.execNetworkConfig.emplace(CONFIG_KEY(GPU_THROUGHPUT_STREAMS), "1");
        }
    }
    return config;
}

CnnConfig ConfigFactory::getCommonConfig(const std::string& flags_d, const std::string& flags_l,
    const std::string& flags_c, bool flags_pc, uint32_t flags_nireq)
{
    CnnConfig config;

    if (!flags_d.empty()) {
        config.devices = flags_d;
    }

    if (!flags_l.empty()) {
        config.cpuExtensionsPath = flags_l;
    }

    if (!flags_c.empty()) {
        config.clKernelsConfigPath = flags_c;
    }

    config.maxAsyncRequests = flags_nireq;

    /** Per layer metrics **/
    if (flags_pc) {
        config.execNetworkConfig.emplace(CONFIG_KEY(PERF_COUNT), PluginConfigParams::YES);
    }

    return config;
}
