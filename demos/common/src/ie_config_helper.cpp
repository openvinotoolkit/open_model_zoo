// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "samples/ie_config_helper.hpp"

std::map<std::string, std::string> createConfig(const std::string& deviceString,
                                                const std::string& nstreamsString,
                                                const int& nthreads,
                                                bool minLatency) {
    std::map<std::string, std::string> config;

    std::set<std::string> devices;
    for (const std::string& device : parseDevices(deviceString)) {
        devices.insert(device);
    }
    std::map<std::string, unsigned> deviceNstreams = parseValuePerDevice(devices, nstreamsString);
    
    for (auto& device : devices) {
        if (device == "CPU") {  // CPU supports a few special performance-oriented keys
            if (minLatency) {
                config.insert({ CONFIG_KEY(CPU_THROUGHPUT_STREAMS), "1" });
                continue;
            }
            
            // limit threading for CPU portion of inference
            if (nthreads != 0)
                config.insert({ CONFIG_KEY(CPU_THREADS_NUM), std::to_string(nthreads) });

            if (deviceString.find("MULTI") != std::string::npos && devices.find("GPU") != devices.end()) {
                config.insert({ CONFIG_KEY(CPU_BIND_THREAD), CONFIG_VALUE(NO) });
            } else {
                // pin threads for CPU portion of inference
                config.insert({ CONFIG_KEY(CPU_BIND_THREAD), CONFIG_VALUE(YES) });
            }

            // for CPU execution, more throughput-oriented execution via streams
            config.insert({ CONFIG_KEY(CPU_THROUGHPUT_STREAMS), (deviceNstreams.count(device) > 0
                                                                 ? std::to_string(deviceNstreams.at(device))
                                                                 : CONFIG_VALUE(CPU_THROUGHPUT_AUTO)) });
        } else if (device == "GPU") {
            if (minLatency) {
                config.insert({ CONFIG_KEY(GPU_THROUGHPUT_STREAMS), "1" });
                continue;
            }

            config.insert({ CONFIG_KEY(GPU_THROUGHPUT_STREAMS), (deviceNstreams.count(device) > 0
                                                                 ? std::to_string(deviceNstreams.at(device))
                                                                 : CONFIG_VALUE(GPU_THROUGHPUT_AUTO)) });

            if (deviceString.find("MULTI") != std::string::npos && devices.find("CPU") != devices.end()) {
                // multi-device execution with the CPU + GPU performs best with GPU throttling hint,
                // which releases another CPU thread (that is otherwise used by the GPU driver for active polling)
                config.insert({ CLDNN_CONFIG_KEY(PLUGIN_THROTTLE), "1" });
            }
        } else if (device == "MYRIAD") {
            // if (!nstreamsString.empty()) {}
            // config.insert({ CONFIG_KEY(_THROUGHPUT_STREAMS), "1" });
        }
    }

    return config;
}
