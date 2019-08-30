// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <opencv2/imgproc.hpp>

#include <ie_plugin_config.hpp>

#include <algorithm>
#include <vector>
#include <map>
#include <string>
#include <set>
#include <memory>
#ifdef WITH_EXTENSIONS
#include <ext_list.hpp>
#endif

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

InferenceEngine::Core
LoadInferenceEngine(const std::vector<std::string>& devices,
                    const std::string& custom_cpu_library,
                    const std::string& custom_cldnn_kernels,
                    bool should_use_perf_counter) {
    std::set<std::string> loadedDevices;
    Core ie;

    for (const auto &device : devices) {
        if (loadedDevices.find(device) != loadedDevices.end()) {
            continue;
        }

        std::cout << "Loading device " << device << std::endl;
        std::cout << ie.GetVersions(device) << std::endl;

        /** Load extensions for the CPU device **/
        if ((device.find("CPU") != std::string::npos)) {
#ifdef WITH_EXTENSIONS
            ie.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
#endif
            if (!custom_cpu_library.empty()) {
                // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                auto extension_ptr = make_so_pointer<IExtension>(custom_cpu_library);
                ie.AddExtension(std::static_pointer_cast<IExtension>(extension_ptr), "CPU");
            }
        } else if (!custom_cldnn_kernels.empty()) {
            // Load Extensions for other plugins not CPU
            ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, custom_cldnn_kernels}}, "GPU");
        }

        if (device.find("CPU") != std::string::npos) {
            ie.SetConfig({{PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES}}, "CPU");
        } else if (device.find("GPU") != std::string::npos) {
            ie.SetConfig({{PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES}}, "GPU");
        }

        if (should_use_perf_counter)
            ie.SetConfig({{PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES}});

        loadedDevices.insert(device);
    }

    return ie;
}
