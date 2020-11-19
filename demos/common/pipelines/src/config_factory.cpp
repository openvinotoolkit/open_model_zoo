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

#include "pipelines/config_factory.h"

#include <set>

#include <samples/args_helper.hpp>
#include <samples/common.hpp>
#include <cldnn/cldnn_config.hpp>

using namespace InferenceEngine;

CnnConfig ConfigFactory::getUserConfig(const std::string& d, const std::string& l, const std::string& c, bool pc,
    uint32_t nireq, const std::string& nstreams, uint32_t nthreads) {
    auto config = getCommonConfig(d, l, c, pc, nireq);
    std::set<std::string> devices;
    for (const std::string& device : parseDevices(d)) {
        devices.insert(device);
    }
    std::map<std::string, unsigned> deviceNstreams = parseValuePerDevice(devices, nstreams);
    for (auto& device : devices) {
        if (device == "CPU") {  // CPU supports a few special performance-oriented keys
            // limit threading for CPU portion of inference
            if (nthreads != 0)
                config.execNetworkConfig.emplace(CONFIG_KEY(CPU_THREADS_NUM), std::to_string(nthreads));

            if (d.find("MULTI") != std::string::npos
                && devices.find("GPU") != devices.end()) {
                config.execNetworkConfig.emplace(CONFIG_KEY(CPU_BIND_THREAD), CONFIG_VALUE(NO));
            }
            else {
                // pin threads for CPU portion of inference
                config.execNetworkConfig.emplace(CONFIG_KEY(CPU_BIND_THREAD), CONFIG_VALUE(YES));
            }

            // for CPU execution, more throughput-oriented execution via streams
            config.execNetworkConfig.emplace(CONFIG_KEY(CPU_THROUGHPUT_STREAMS),
                (deviceNstreams.count(device) > 0 ? std::to_string(deviceNstreams.at(device))
                    : CONFIG_VALUE(CPU_THROUGHPUT_AUTO)));
        }
        else if (device == "GPU") {
            config.execNetworkConfig.emplace(CONFIG_KEY(GPU_THROUGHPUT_STREAMS),
                (deviceNstreams.count(device) > 0 ? std::to_string(deviceNstreams.at(device))
                    : CONFIG_VALUE(GPU_THROUGHPUT_AUTO)));

            if (d.find("MULTI") != std::string::npos
                && devices.find("CPU") != devices.end()) {
                // multi-device execution with the CPU + GPU performs best with GPU throttling hint,
                // which releases another CPU thread (that is otherwise used by the GPU driver for active polling)
                config.execNetworkConfig.emplace(CLDNN_CONFIG_KEY(PLUGIN_THROTTLE), "1");
            }
        }
    }
    return config;
}

CnnConfig ConfigFactory::getMinLatencyConfig(const std::string& d, const std::string& l, const std::string& c, bool pc, uint32_t nireq) {
    auto config = getCommonConfig(d, l, c, pc, nireq);
    std::set<std::string> devices;
    for (const std::string& device : parseDevices(d)) {
        devices.insert(device);
    }
    for (auto& device : devices) {
        if (device == "CPU") {  // CPU supports a few special performance-oriented keys
            config.execNetworkConfig.emplace(CONFIG_KEY(CPU_THROUGHPUT_STREAMS), "1");
        }
        else if (device == "GPU") {
            config.execNetworkConfig.emplace(CONFIG_KEY(GPU_THROUGHPUT_STREAMS), "1");
        }
    }
    return config;
}

CnnConfig ConfigFactory::getCommonConfig(const std::string& d, const std::string& l, const std::string& c, bool pc, uint32_t nireq) {
    CnnConfig config;

    if (!d.empty()) {
        config.devices = d;
    }

    if (!l.empty()) {
        config.cpuExtensionsPath = l;
    }

    if (!c.empty()) {
        config.clKernelsConfigPath = c;
    }

    if (nireq) {
        config.maxAsyncRequests = nireq;
    }

    /** Per layer metrics **/
    if (pc) {
        config.execNetworkConfig.emplace(CONFIG_KEY(PERF_COUNT), PluginConfigParams::YES);
    }

    return config;
}
