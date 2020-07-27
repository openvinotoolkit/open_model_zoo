#include "config_factory.h"

#include <set>

#include <samples/args_helper.hpp>
#include <samples/common.hpp>
#include <cldnn/cldnn_config.hpp>
#include "command_parameters.h"

using namespace InferenceEngine;

CnnConfig ConfigFactory::GetUserConfig() {
    auto config = GetCommonConfig();
    std::set<std::string> devices;
    for (const std::string& device : parseDevices(FLAGS_d)) {
        devices.insert(device);
    }
    std::map<std::string, unsigned> deviceNstreams = parseValuePerDevice(devices, FLAGS_nstreams);
    for (auto & device : devices) {
        if (device == "CPU") {  // CPU supports a few special performance-oriented keys
            // limit threading for CPU portion of inference
            if (FLAGS_nthreads != 0)
                config.execNetworkConfig.emplace(CONFIG_KEY(CPU_THREADS_NUM), std::to_string(FLAGS_nthreads));

            if (FLAGS_d.find("MULTI") != std::string::npos
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

            if (FLAGS_d.find("MULTI") != std::string::npos
                && devices.find("CPU") != devices.end()) {
                // multi-device execution with the CPU + GPU performs best with GPU throttling hint,
                // which releases another CPU thread (that is otherwise used by the GPU driver for active polling)
                config.execNetworkConfig.emplace(CLDNN_CONFIG_KEY(PLUGIN_THROTTLE), "1");
            }
        }
    }
    return config;
}

CnnConfig ConfigFactory::GetMinLatencyConfig() {
    auto config = GetCommonConfig();
    std::set<std::string> devices;
    for (const std::string& device : parseDevices(FLAGS_d)) {
        devices.insert(device);
    }
    std::map<std::string, unsigned> deviceNstreams = parseValuePerDevice(devices, FLAGS_nstreams);
    for (auto & device : devices) {
        if (device == "CPU") {  // CPU supports a few special performance-oriented keys
            config.execNetworkConfig.emplace(CONFIG_KEY(CPU_THROUGHPUT_STREAMS), "1");
        }
        else if (device == "GPU") {
            config.execNetworkConfig.emplace(CONFIG_KEY(GPU_THROUGHPUT_STREAMS), "1");
        }
    }
    return config;
}

CnnConfig ConfigFactory::GetCommonConfig() {
    CnnConfig config;

    if (!FLAGS_d.empty()) {
        config.devices = FLAGS_d;
    }

    if (!FLAGS_l.empty()) {
        config.cpuExtensionsPath = FLAGS_l;
    }

    if (!FLAGS_c.empty()) {
        config.clKernelsConfigPath = FLAGS_c;
    }

    if (FLAGS_nireq) {
        config.maxAsyncRequests = FLAGS_nireq;
    }

    /** Per layer metrics **/
    if (FLAGS_pc) {
        config.execNetworkConfig.emplace(CONFIG_KEY(PERF_COUNT), PluginConfigParams::YES);
    }

    return config;
}
