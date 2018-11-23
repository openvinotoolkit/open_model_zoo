/*
// Copyright (c) 2018 Intel Corporation
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

#include "utils.hpp"

#include <opencv2/imgproc.hpp>

#include <ie_plugin_config.hpp>

#include <algorithm>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <ext_list.hpp>

using namespace InferenceEngine;

namespace {
template <typename StreamType>
void SaveDetectionLogToStream(StreamType& stream,
                              const DetectionLog& log) {
    for (const auto& entry : log) {
        std::vector<TrackedObject> objects(entry.objects.begin(),
                                           entry.objects.end());
        std::sort(objects.begin(), objects.end(),
                  [](const TrackedObject& a,
                     const TrackedObject& b)
                  { return a.object_id < b.object_id; });
        for (const auto& object : objects) {
            auto frame_idx_to_save = entry.frame_idx;
            stream << frame_idx_to_save << ',';
            stream << object.object_id << ','
                << object.rect.x << ',' << object.rect.y << ','
                << object.rect.width << ',' << object.rect.height;
            stream << '\n';
        }
    }
}
}  // anonymous namespace

void DrawPolyline(const std::vector<cv::Point>& polyline,
                  const cv::Scalar& color, cv::Mat* image, int lwd) {
    PT_CHECK(image);
    PT_CHECK(!image->empty());
    PT_CHECK_EQ(image->type(), CV_8UC3);
    PT_CHECK_GT(lwd, 0);
    PT_CHECK_LT(lwd, 20);

    for (size_t i = 1; i < polyline.size(); i++) {
        cv::line(*image, polyline[i - 1], polyline[i], color, lwd);
    }
}

void SaveDetectionLogToTrajFile(const std::string& path,
                                const DetectionLog& log) {
    std::ofstream file(path.c_str());
    PT_CHECK(file.is_open());
    SaveDetectionLogToStream(file, log);
}

void PrintDetectionLog(const DetectionLog& log) {
    SaveDetectionLogToStream(std::cout, log);
}


std::map<std::string, InferencePlugin>
LoadPluginForDevices(const std::vector<std::string>& devices,
                     const std::string& custom_cpu_library,
                     const std::string& custom_cldnn_kernels,
                     bool should_use_perf_counter) {
    std::map<std::string, InferencePlugin> plugins_for_devices;

    for (const auto &device : devices) {
        if (plugins_for_devices.find(device) != plugins_for_devices.end()) {
            continue;
        }
        std::cout << "Loading plugin " << device << std::endl;
        InferencePlugin plugin = PluginDispatcher({"../../../lib/intel64", ""}).getPluginByDevice(device);
        printPluginVersion(plugin, std::cout);
        /** Load extensions for the CPU plugin **/
        if ((device.find("CPU") != std::string::npos)) {
            plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
            if (!custom_cpu_library.empty()) {
                // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                auto extension_ptr = make_so_pointer<IExtension>(custom_cpu_library);
                plugin.AddExtension(std::static_pointer_cast<IExtension>(extension_ptr));
            }
        } else if (!custom_cldnn_kernels.empty()) {
            // Load Extensions for other plugins not CPU
            plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE,
                             custom_cldnn_kernels}});
        }
        plugin.SetConfig({{PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES}});
        if (should_use_perf_counter)
            plugin.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});
        plugins_for_devices[device] = plugin;
    }
    return plugins_for_devices;
}

