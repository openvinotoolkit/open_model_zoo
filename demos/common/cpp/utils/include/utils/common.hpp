// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with common samples functionality
 * @file common.hpp
 */

#pragma once

#include <string>
#include <map>
#include <vector>
#include <list>
#include <limits>
#include <functional>
#include <fstream>
#include <iomanip>
#include <utility>
#include <algorithm>
#include <random>
#include <iostream>

#include <inference_engine.hpp>

#ifndef UNUSED
  #ifdef _WIN32
    #define UNUSED
  #else
    #define UNUSED  __attribute__((unused))
  #endif
#endif

template <typename T, std::size_t N>
constexpr std::size_t arraySize(const T (&)[N]) noexcept {
    return N;
}

// Helpers to print IE version information.
// We don't directly define operator<< for InferenceEngine::Version
// and such, because that won't get picked up by argument-dependent lookup
// due to not being in the same namespace as the Version class itself.
// We need ADL to work in order to print these objects using slog.
// So instead, we define wrapper classes and operator<< for those classes.

class PrintableIeVersion {
public:
    using ref_type = const InferenceEngine::Version &;

    PrintableIeVersion(ref_type version) : version(version) {}

    friend std::ostream &operator<<(std::ostream &os, const PrintableIeVersion &p) {
        ref_type version = p.version;

        return os << "\t" << version.description << " version ......... "
           << IE_VERSION_MAJOR << "." << IE_VERSION_MINOR
           << "\n\tBuild ........... " << IE_VERSION_PATCH;
    }

private:
    ref_type version;
};

inline PrintableIeVersion printable(PrintableIeVersion::ref_type version) {
    return { version };
}

class PrintableIeVersionMap {
public:
    using ref_type = const std::map<std::string, InferenceEngine::Version> &;

    PrintableIeVersionMap(ref_type versions) : versions(versions) {}

    friend std::ostream &operator<<(std::ostream &os, const PrintableIeVersionMap &p) {
        ref_type versions = p.versions;

        for (const auto &version : versions) {
            os << "\t" << version.first << std::endl
               << printable(version.second) << std::endl;
        }

        return os;
    }

private:
    ref_type versions;
};

inline PrintableIeVersionMap printable(PrintableIeVersionMap::ref_type versions) {
    return { versions };
}

/**
 * @class Color
 * @brief A Color class stores channels of a given color
 */
class Color {
private:
    unsigned char _r;
    unsigned char _g;
    unsigned char _b;

public:
    /**
     * A default constructor.
     * @param r - value for red channel
     * @param g - value for green channel
     * @param b - value for blue channel
     */
    Color(unsigned char r,
          unsigned char g,
          unsigned char b) : _r(r), _g(g), _b(b) {}

    inline unsigned char red() const {
        return _r;
    }

    inline unsigned char blue() const {
        return _b;
    }

    inline unsigned char green() const {
        return _g;
    }
};

// Known colors for training classes from the Cityscapes dataset
static UNUSED const Color CITYSCAPES_COLORS[] = {
    { 128, 64,  128 },
    { 232, 35,  244 },
    { 70,  70,  70 },
    { 156, 102, 102 },
    { 153, 153, 190 },
    { 153, 153, 153 },
    { 30,  170, 250 },
    { 0,   220, 220 },
    { 35,  142, 107 },
    { 152, 251, 152 },
    { 180, 130, 70 },
    { 60,  20,  220 },
    { 0,   0,   255 },
    { 142, 0,   0 },
    { 70,  0,   0 },
    { 100, 60,  0 },
    { 90,  0,   0 },
    { 230, 0,   0 },
    { 32,  11,  119 },
    { 0,   74,  111 },
    { 81,  0,   81 }
};

static std::vector<std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo>>
perfCountersSorted(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap) {
    using perfItem = std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo>;
    std::vector<perfItem> sorted;
    for (auto &kvp : perfMap) sorted.push_back(kvp);

    std::stable_sort(sorted.begin(), sorted.end(),
                     [](const perfItem& l, const perfItem& r) {
                         return l.second.execution_index < r.second.execution_index;
                     });

    return sorted;
}

static UNUSED void printPerformanceCounts(const std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& performanceMap,
                                          std::ostream &stream, const std::string &deviceName,
                                          bool bshowHeader = true) {
    long long totalTime = 0;
    // Print performance counts
    if (bshowHeader) {
        stream << std::endl << "performance counts:" << std::endl << std::endl;
    }

    auto performanceMapSorted = perfCountersSorted(performanceMap);

    for (const auto & it : performanceMapSorted) {
        std::string toPrint(it.first);
        const int maxLayerName = 30;

        if (it.first.length() >= maxLayerName) {
            toPrint  = it.first.substr(0, maxLayerName - 4);
            toPrint += "...";
        }


        stream << std::setw(maxLayerName) << std::left << toPrint;
        switch (it.second.status) {
        case InferenceEngine::InferenceEngineProfileInfo::EXECUTED:
            stream << std::setw(15) << std::left << "EXECUTED";
            break;
        case InferenceEngine::InferenceEngineProfileInfo::NOT_RUN:
            stream << std::setw(15) << std::left << "NOT_RUN";
            break;
        case InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT:
            stream << std::setw(15) << std::left << "OPTIMIZED_OUT";
            break;
        }
        stream << std::setw(30) << std::left << "layerType: " + std::string(it.second.layer_type) + " ";
        stream << std::setw(20) << std::left << "realTime: " + std::to_string(it.second.realTime_uSec);
        stream << std::setw(20) << std::left << "cpu: "  + std::to_string(it.second.cpu_uSec);
        stream << " execType: " << it.second.exec_type << std::endl;
        if (it.second.realTime_uSec > 0) {
            totalTime += it.second.realTime_uSec;
        }
    }
    stream << std::setw(20) << std::left << "Total time: " + std::to_string(totalTime) << " microseconds" << std::endl;
    std::cout << std::endl;
    std::cout << "Full device name: " << deviceName << std::endl;
    std::cout << std::endl;
}

static UNUSED void printPerformanceCounts(InferenceEngine::InferRequest request, std::ostream &stream, std::string deviceName, bool bshowHeader = true) {
    auto performanceMap = request.GetPerformanceCounts();
    printPerformanceCounts(performanceMap, stream, deviceName, bshowHeader);
}

inline std::map<std::string, std::string> getMapFullDevicesNames(InferenceEngine::Core& ie, std::vector<std::string> devices) {
    std::map<std::string, std::string> devicesMap;
    InferenceEngine::Parameter p;
    for (std::string& deviceName : devices) {
        if (deviceName != "") {
            try {
                p = ie.GetMetric(deviceName, METRIC_KEY(FULL_DEVICE_NAME));
                devicesMap.insert(std::pair<std::string, std::string>(deviceName, p.as<std::string>()));
            }
            catch (InferenceEngine::Exception &) {
            }
        }
    }
    return devicesMap;
}

inline std::string getFullDeviceName(std::map<std::string, std::string>& devicesMap, std::string device) {
    std::map<std::string, std::string>::iterator it = devicesMap.find(device);
    if (it != devicesMap.end()) {
        return it->second;
    } else {
        return "";
    }
}

inline std::string getFullDeviceName(InferenceEngine::Core& ie, std::string device) {
    InferenceEngine::Parameter p;
    try {
        p = ie.GetMetric(device, METRIC_KEY(FULL_DEVICE_NAME));
        return  p.as<std::string>();
    }
    catch (InferenceEngine::Exception &) {
        return "";
    }
}

inline std::size_t getTensorWidth(const InferenceEngine::TensorDesc& desc) {
    const auto& layout = desc.getLayout();
    const auto& dims = desc.getDims();
    const auto& size = dims.size();
    if ((size >= 2) &&
        (layout == InferenceEngine::Layout::NCHW  ||
         layout == InferenceEngine::Layout::NHWC  ||
         layout == InferenceEngine::Layout::NCDHW ||
         layout == InferenceEngine::Layout::NDHWC ||
         layout == InferenceEngine::Layout::OIHW  ||
         layout == InferenceEngine::Layout::CHW   ||
         layout == InferenceEngine::Layout::HW)) {
        // Regardless of layout, dimensions are stored in fixed order
        return dims.back();
    } else {
        throw std::runtime_error("Tensor does not have width dimension");
    }
    return 0;
}

inline std::size_t getTensorHeight(const InferenceEngine::TensorDesc& desc) {
    const auto& layout = desc.getLayout();
    const auto& dims = desc.getDims();
    const auto& size = dims.size();
    if ((size >= 2) &&
        (layout == InferenceEngine::Layout::NCHW  ||
         layout == InferenceEngine::Layout::NHWC  ||
         layout == InferenceEngine::Layout::NCDHW ||
         layout == InferenceEngine::Layout::NDHWC ||
         layout == InferenceEngine::Layout::OIHW  ||
         layout == InferenceEngine::Layout::CHW   ||
         layout == InferenceEngine::Layout::HW)) {
        // Regardless of layout, dimensions are stored in fixed order
        return dims.at(size - 2);
    } else {
        throw std::runtime_error("Tensor does not have height dimension");
    }
    return 0;
}

inline std::size_t getTensorChannels(const InferenceEngine::TensorDesc& desc) {
    const auto& layout = desc.getLayout();
    if (layout == InferenceEngine::Layout::NCHW  ||
        layout == InferenceEngine::Layout::NHWC  ||
        layout == InferenceEngine::Layout::NCDHW ||
        layout == InferenceEngine::Layout::NDHWC ||
        layout == InferenceEngine::Layout::C     ||
        layout == InferenceEngine::Layout::CHW   ||
        layout == InferenceEngine::Layout::NC    ||
        layout == InferenceEngine::Layout::CN) {
        // Regardless of layout, dimensions are stored in fixed order
        const auto& dims = desc.getDims();
        switch (desc.getLayoutByDims(dims)) {
            case InferenceEngine::Layout::C:     return dims.at(0);
            case InferenceEngine::Layout::NC:    return dims.at(1);
            case InferenceEngine::Layout::CHW:   return dims.at(0);
            case InferenceEngine::Layout::NCHW:  return dims.at(1);
            case InferenceEngine::Layout::NCDHW: return dims.at(1);
            case InferenceEngine::Layout::SCALAR:   // [[fallthrough]]
            case InferenceEngine::Layout::BLOCKED:  // [[fallthrough]]
            default:
                throw std::runtime_error("Tensor does not have channels dimension");
        }
    } else {
        throw std::runtime_error("Tensor does not have channels dimension");
    }
    return 0;
}

inline std::size_t getTensorBatch(const InferenceEngine::TensorDesc& desc) {
    const auto& layout = desc.getLayout();
    if (layout == InferenceEngine::Layout::NCHW  ||
        layout == InferenceEngine::Layout::NHWC  ||
        layout == InferenceEngine::Layout::NCDHW ||
        layout == InferenceEngine::Layout::NDHWC ||
        layout == InferenceEngine::Layout::NC    ||
        layout == InferenceEngine::Layout::CN) {
        // Regardless of layout, dimensions are stored in fixed order
        const auto& dims = desc.getDims();
        switch (desc.getLayoutByDims(dims)) {
            case InferenceEngine::Layout::NC:    return dims.at(0);
            case InferenceEngine::Layout::NCHW:  return dims.at(0);
            case InferenceEngine::Layout::NCDHW: return dims.at(0);
            case InferenceEngine::Layout::CHW:      // [[fallthrough]]
            case InferenceEngine::Layout::C:        // [[fallthrough]]
            case InferenceEngine::Layout::SCALAR:   // [[fallthrough]]
            case InferenceEngine::Layout::BLOCKED:  // [[fallthrough]]
            default:
                throw std::runtime_error("Tensor does not have channels dimension");
        }
    } else {
        throw std::runtime_error("Tensor does not have channels dimension");
    }
    return 0;
}

inline void showAvailableDevices() {
    InferenceEngine::Core ie;
    std::vector<std::string> devices = ie.GetAvailableDevices();

    std::cout << std::endl;
    std::cout << "Available target devices:";
    for (const auto& device : devices) {
        std::cout << "  " << device;
    }
    std::cout << std::endl;
}

inline std::string fileNameNoExt(const std::string &filepath) {
    auto pos = filepath.rfind('.');
    if (pos == std::string::npos) return filepath;
    return filepath.substr(0, pos);
}
