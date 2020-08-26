// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "samples/ie_config_helper.hpp"
#include "samples/args_helper.hpp"

std::string formatDeviceString(const std::string& deviceString) {
    std::string formattedString = deviceString;
    bool preserveCase = false;

    for (size_t i = 0; i < deviceString.size(); ++i) {
        // These two conditions handle the special case of MYRIAD device names in "ma1234" format, where letters should
        // not be transformed to upper case.
        // if (deviceString[i] == 'm' && i+1 != deviceString.size() && deviceString[i+1] == 'a') {
        //     preserveCase = true;
        // }
        if (preserveCase && deviceString[i] == ',') {
            preserveCase = false;
        }

        if (!preserveCase) {
            formattedString[i] = std::toupper(deviceString[i]);
        }
    }

    return formattedString;
}

std::map<std::string, std::string> createConfig(const std::string& deviceString,
                                                const std::string& nstreamsString,
                                                int nthreads,
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
        } else if (device.find("MYRIAD") == 0 /*|| device.find("ma") == 0*/) {
            if (minLatency) {
                config.insert({ InferenceEngine::MYRIAD_THROUGHPUT_STREAMS, "1" });
                continue;
            }

            if (deviceNstreams.count(device) > 0) {
                config.insert({ InferenceEngine::MYRIAD_THROUGHPUT_STREAMS,
                                std::to_string(deviceNstreams.at(device)) });
            }
        }
    }

    return config;
}
