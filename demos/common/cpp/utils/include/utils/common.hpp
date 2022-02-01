// Copyright (C) 2018-2022 Intel Corporation
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
#include "openvino/openvino.hpp"
#include "utils/slog.hpp"
#include "utils/args_helper.hpp"

#ifndef UNUSED
  #ifdef _WIN32
    #define UNUSED
  #else
    #define UNUSED  __attribute__((unused))
  #endif
#endif

template <typename T, std::size_t N>
constexpr std::size_t arraySize(const T(&)[N]) noexcept {
    return N;
}

static inline void catcher() noexcept {
    if (std::current_exception()) {
        try {
            std::rethrow_exception(std::current_exception());
        } catch (const std::exception& error) {
            slog::err << error.what() << slog::endl;
        } catch (...) {
            slog::err << "Non-exception object thrown" << slog::endl;
        }
        std::exit(1);
    }
    std::abort();
}

template <typename T>
T clamp(T value, T low, T high) {
    return value < low ? low : (value > high ? high : value);
}

// Redefine operator<< for LogStream to print IE version information.
inline slog::LogStream& operator<<(slog::LogStream& os, const InferenceEngine::Version& version) {
    os << "OpenVINO Inference Engine" << slog::endl;
    os << "\tversion: " << IE_VERSION_MAJOR << "." << IE_VERSION_MINOR << "." << IE_VERSION_PATCH << slog::endl;
    os << "\tbuild: " << version.buildNumber;

    return os;
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
#if defined(OV_NEW_API)
    ov::Core core;
    std::vector<std::string> devices = core.get_available_devices();
#else
    InferenceEngine::Core ie;
    std::vector<std::string> devices = ie.GetAvailableDevices();
#endif

    std::cout << std::endl;
    std::cout << "Available target devices:";
    for (const auto& device : devices) {
        std::cout << "  " << device;
    }
    std::cout << std::endl;
}

inline std::string fileNameNoExt(const std::string& filepath) {
    auto pos = filepath.rfind('.');
    if (pos == std::string::npos) return filepath;
    return filepath.substr(0, pos);
}

inline void logExecNetworkInfo(const InferenceEngine::ExecutableNetwork& execNetwork, const std::string& modelName,
    const std::string& deviceName, const std::string& modelType = "") {
    slog::info << "The " << modelType << (modelType.empty() ? "" : " ") << "model " << modelName << " is loaded to " << deviceName << slog::endl;
    std::set<std::string> devices;
    for (const std::string& device : parseDevices(deviceName)) {
        devices.insert(device);
    }

    if (devices.find("AUTO") == devices.end()) { // do not print info for AUTO device
        for (const auto& device : devices) {
            try {
                slog::info << "\tDevice: " << device << slog::endl;
                std::string nstreams = execNetwork.GetConfig(device + "_THROUGHPUT_STREAMS").as<std::string>();
                slog::info << "\t\tNumber of streams: " << nstreams << slog::endl;
                if (device == "CPU") {
                    std::string nthreads = execNetwork.GetConfig("CPU_THREADS_NUM").as<std::string>();
                    slog::info << "\t\tNumber of threads: " << (nthreads == "0" ? "AUTO" : nthreads) << slog::endl;
                }
            }
            catch (const InferenceEngine::Exception&) {}
        }
    }
}

inline
void logCompiledModelInfo(
    const ov::CompiledModel& compiledModel,
    const std::string& modelName,
    const std::string& deviceName,
    const std::string& modelType = "")
{
    slog::info << "The " << modelType << (modelType.empty() ? "" : " ") << "model " << modelName << " is loaded to " << deviceName << slog::endl;
    std::set<std::string> devices;
    for (const std::string& device : parseDevices(deviceName)) {
        devices.insert(device);
    }

    if (devices.find("AUTO") == devices.end()) { // do not print info for AUTO device
        for (const auto& device : devices) {
            try {
                slog::info << "\tDevice: " << device << slog::endl;
                std::string nstreams = compiledModel.get_property(device + "_THROUGHPUT_STREAMS").as<std::string>();
                slog::info << "\t\tNumber of streams: " << nstreams << slog::endl;
                if (device == "CPU") {
                    std::string nthreads = compiledModel.get_property("CPU_THREADS_NUM").as<std::string>();
                    slog::info << "\t\tNumber of threads: " << (nthreads == "0" ? "AUTO" : nthreads) << slog::endl;
                }
            }
            catch (const ov::Exception&) {}
        }
    }
}

inline
void logBasicModelInfo(const std::shared_ptr<ov::Model>& model) {
    slog::info << "model name: " << model->get_friendly_name() << slog::endl;

    // Dump information about model inputs/outputs
    ov::OutputVector inputs = model->inputs();
    ov::OutputVector outputs = model->outputs();

    slog::info << "inputs: " << slog::endl;
    for (const ov::Output<ov::Node> input : inputs)
    {
        const std::string name = input.get_any_name();
        const ov::element::Type type = input.get_element_type();
        const ov::PartialShape shape = input.get_partial_shape();

        slog::info << name << ", " << type << ", " << shape << slog::endl;
    }

    slog::info << "outputs: " << slog::endl;
    for (const ov::Output<ov::Node> output : outputs)
    {
        const std::string name = output.get_any_name();
        const ov::element::Type type = output.get_element_type();
        const ov::PartialShape shape = output.get_partial_shape();

        slog::info << name << ", " << type << ", " << shape << slog::endl;
    }

    return;
}
