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

/**
 * @brief a header file with common demos functionality
 * @file common.hpp
 */

#pragma once

#include <string>
#include <map>
#include <vector>
#include <limits>
#include <random>
#include <cctype>
#include <functional>
#include <time.h>
#include <iostream>
#include <iomanip>

#include <algorithm>
#include <chrono>

#include <ie_plugin_dispatcher.hpp>
#include <ie_plugin_ptr.hpp>
#include <ie_device.hpp>

#ifndef UNUSED
  #ifdef WIN32
    #define UNUSED
  #else
    #define UNUSED  __attribute__((unused))
  #endif
#endif

/**
* @brief name a layer extension plugin filename with CVSDK env, for mklDNN
* @return layer extension plugin file name as a string
*/
std::string GetCPUExtensionPluginFilename() {
    std::string sfilename = "";

    const char * val = getenv("INTEL_CVSDK_DIR");

    if (val != NULL) {
        std::string s_val(val);

#ifdef WIN32
        sfilename = s_val + "\\deployment_tools\\inference_engine\\bin\\intel64\\release\\cpu_extension.dll";
#else
        sfilename = s_val + "/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension.so";
#endif
    }

    return sfilename;
}

/**
 * @brief Trims from both ends (in place)
 * @param s - string to trim
 * @return trimmed string
 */
inline std::string &trim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}

/**
* @brief Converts string to TargetDevice
* @param deviceName - string value representing device
* @return TargetDevice value that corresponds to input string.
*         eDefault in case no corresponding value was found
*/
static InferenceEngine::TargetDevice getDeviceFromStr(const std::string &deviceName) {
    static std::map<std::string, InferenceEngine::TargetDevice> deviceFromNameMap = {
        { "CPU", InferenceEngine::TargetDevice::eCPU },
        { "GPU", InferenceEngine::TargetDevice::eGPU },
        { "BALANCED", InferenceEngine::TargetDevice::eBalanced }
    };

    auto val = deviceFromNameMap.find(deviceName);
    return val != deviceFromNameMap.end() ? val->second : InferenceEngine::TargetDevice::eDefault;
}

/**
* @brief Loads plugin from directories
* @param pluginDirs - plugin paths
* @param plugin - plugin name
* @param device - device to infer on
* @return Plugin pointer
*/
static InferenceEngine::InferenceEnginePluginPtr selectPlugin(const std::vector<std::string> &pluginDirs,
                                                              const std::string &plugin,
                                                              InferenceEngine::TargetDevice device) {
    InferenceEngine::PluginDispatcher dispatcher(pluginDirs);

    if (!plugin.empty()) {
        return dispatcher.getPluginByName(plugin);
    } else {
        return dispatcher.getSuitablePlugin(device);
    }
}

/**
 * @brief Loads plugin from directories
 * @param pluginDirs - plugin paths
 * @param plugin - plugin name
 * @param device - string representation of device to infer on
 * @return Plugin pointer
 */
static UNUSED InferenceEngine::InferenceEnginePluginPtr selectPlugin(const std::vector<std::string> &pluginDirs,
                                                                     const std::string &plugin,
                                                                     const std::string &device) {
    return selectPlugin(pluginDirs, plugin, getDeviceFromStr(device));
}

/**
 * @brief Gets filename without extension
 * @param filepath - full file name
 * @return filename without extension
 */
static UNUSED std::string fileNameNoExt(const std::string &filepath) {
    auto pos = filepath.rfind('.');
    if (pos == std::string::npos) return filepath;
    return filepath.substr(0, pos);
}

/**
* @brief Get extension from filename
* @param filename - name of the file which extension should be extracted
* @return string with extracted file extension
*/
inline std::string fileExt(const std::string& filename) {
    auto pos = filename.rfind('.');
    if (pos == std::string::npos) return "";
    return filename.substr(pos + 1);
}

namespace InferenceEngine {
namespace details {

/**
* @brief vector serialisation to be used in exception
     * @param out - stream to write values to
     * @param vec - vector to print values from
*/
    template<typename T>
    inline std::ostream &operator<<(std::ostream &out, const std::vector<T> &vec) {
    if (vec.empty()) return std::operator<<(out, "[]");
    out << "[" << vec[0];
        for (unsigned i = 1; i < vec.size(); i++) {
        out << ", " << vec[i];
    }
    return out << "]";
}

}  // namespace details
}  // namespace InferenceEngine


static UNUSED std::ostream &operator<<(std::ostream &os, const InferenceEngine::Version *version) {
    os << "\n\tAPI version ............ ";
    if (nullptr == version) {
        os << "UNKNOWN";
    } else {
        os << version->apiVersion.major << "." << version->apiVersion.minor;
        if (nullptr != version->buildNumber) {
            os << "\n\t" << "Build .................. " << version->buildNumber;
        }
        if (nullptr != version->description) {
            os << "\n\t" << "Description ....... " << version->description;
        }
    }
    return os;
}

/**
 * @class PluginVersion
 * @brief A PluginVersion class stores plugin version and initialization status
 */
struct PluginVersion : public InferenceEngine::Version {
    bool initialized = false;

    explicit PluginVersion(const InferenceEngine::Version *ver) {
        if (nullptr == ver) {
            return;
        }
        InferenceEngine::Version::operator=(*ver);
        initialized = true;
    }

    operator bool() const noexcept {
        return initialized;
    }
};

static UNUSED std::ostream &operator<<(std::ostream &os, const PluginVersion &version) {
    os << "\tPlugin version ......... ";
    if (!version) {
        os << "UNKNOWN";
    } else {
        os << version.apiVersion.major << "." << version.apiVersion.minor;
    }

    os << "\n\tPlugin name ............ ";
    if (!version || version.description == nullptr) {
        os << "UNKNOWN";
    } else {
        os << version.description;
    }

    os << "\n\tPlugin build ........... ";
    if (!version || version.buildNumber == nullptr) {
        os << "UNKNOWN";
    } else {
        os << version.buildNumber;
    }

    return os;
}

inline void printPluginVersion(InferenceEngine::InferenceEnginePluginPtr ptr, std::ostream& stream) {
    const PluginVersion *pluginVersion = nullptr;
    ptr->GetVersion((const InferenceEngine::Version*&)pluginVersion);
    stream << pluginVersion << std::endl;
}

static UNUSED std::vector<std::vector<size_t>> blobToImageOutputArray(InferenceEngine::TBlob<float>::Ptr output,
                                                                      size_t *pWidth, size_t *pHeight,
                                                                      size_t *pChannels) {
    std::vector<std::vector<size_t>> outArray;
    size_t W = output->dims().at(0);
    size_t H = output->dims().at(1);
    size_t C = output->dims().at(2);

    // Get classes
    const float *outData = output->data();
    for (unsigned h = 0; h < H; h++) {
        std::vector<size_t> row;
        for (unsigned w = 0; w < W; w++) {
            float max_value = outData[h * W + w];
            size_t index = 0;
            for (size_t c = 1; c < C; c++) {
                size_t dataIndex = c * H * W + h * W + w;
                if (outData[dataIndex] > max_value) {
                    index = c;
                    max_value = outData[dataIndex];
                }
            }
            row.push_back(index);
        }
        outArray.push_back(row);
    }

    if (pWidth != nullptr) *pWidth = W;
    if (pHeight != nullptr) *pHeight = H;
    if (pChannels != nullptr) *pChannels = C;

    return outArray;
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

    inline unsigned char red() {
        return _r;
    }

    inline unsigned char blue() {
        return _b;
    }

    inline unsigned char green() {
        return _g;
    }
};

/**
 * @brief Writes output data to image
 * @param name - image name
 * @param data - output data
 * @param classesNum - the number of classes
 * @return false if error else true
 */
static UNUSED void writeOutputBmp(std::vector<std::vector<size_t>> data, size_t classesNum, std::ostream &outFile) {
    unsigned int seed = (unsigned int) time(NULL);
    // Known colors for training classes from Cityscape dataset
    std::vector<Color> colors = {
        {128, 64,  128},
        {232, 35,  244},
        {70,  70,  70},
        {156, 102, 102},
        {153, 153, 190},
        {153, 153, 153},
        {30,  170, 250},
        {0,   220, 220},
        {35,  142, 107},
        {152, 251, 152},
        {180, 130, 70},
        {60,  20,  220},
        {0,   0,   255},
        {142, 0,   0},
        {70,  0,   0},
        {100, 60,  0},
        {90,  0,   0},
        {230, 0,   0},
        {32,  11,  119},
        {0,   74,  111},
        {81,  0,   81}
    };

    while (classesNum > colors.size()) {
        static std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(0, 255);
        Color color(dist(rng), dist(rng), dist(rng));
        colors.push_back(color);
    }

    unsigned char file[14] = {
            'B', 'M',           // magic
            0, 0, 0, 0,         // size in bytes
            0, 0,               // app data
            0, 0,               // app data
            40 + 14, 0, 0, 0      // start of data offset
    };
    unsigned char info[40] = {
            40, 0, 0, 0,        // info hd size
            0, 0, 0, 0,         // width
            0, 0, 0, 0,         // height
            1, 0,               // number color planes
            24, 0,              // bits per pixel
            0, 0, 0, 0,         // compression is none
            0, 0, 0, 0,         // image bits size
            0x13, 0x0B, 0, 0,   // horz resoluition in pixel / m
            0x13, 0x0B, 0, 0,   // vert resolutions (0x03C3 = 96 dpi, 0x0B13 = 72 dpi)
            0, 0, 0, 0,         // #colors in pallete
            0, 0, 0, 0,         // #important colors
    };

    auto height = data.size();
    auto width = data.at(0).size();

    if (height > (size_t) std::numeric_limits<int32_t>::max || width > (size_t) std::numeric_limits<int32_t>::max) {
        THROW_IE_EXCEPTION << "File size is too big: " << height << " X " << width;
    }

    int padSize = static_cast<int>(4 - (width * 3) % 4) % 4;
    int sizeData = static_cast<int>(width * height * 3 + height * padSize);
    int sizeAll = sizeData + sizeof(file) + sizeof(info);

    file[2] = (unsigned char) (sizeAll);
    file[3] = (unsigned char) (sizeAll >> 8);
    file[4] = (unsigned char) (sizeAll >> 16);
    file[5] = (unsigned char) (sizeAll >> 24);

    info[4] = (unsigned char) (width);
    info[5] = (unsigned char) (width >> 8);
    info[6] = (unsigned char) (width >> 16);
    info[7] = (unsigned char) (width >> 24);

    int32_t negativeHeight = -(int32_t) height;
    info[8] = (unsigned char) (negativeHeight);
    info[9] = (unsigned char) (negativeHeight >> 8);
    info[10] = (unsigned char) (negativeHeight >> 16);
    info[11] = (unsigned char) (negativeHeight >> 24);

    info[20] = (unsigned char) (sizeData);
    info[21] = (unsigned char) (sizeData >> 8);
    info[22] = (unsigned char) (sizeData >> 16);
    info[23] = (unsigned char) (sizeData >> 24);

    outFile.write(reinterpret_cast<char *>(file), sizeof(file));
    outFile.write(reinterpret_cast<char *>(info), sizeof(info));

    unsigned char pad[3] = {0, 0, 0};

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            unsigned char pixel[3];
            size_t index = data.at(y).at(x);
            pixel[0] = colors.at(index).red();
            pixel[1] = colors.at(index).green();
            pixel[2] = colors.at(index).blue();
            outFile.write(reinterpret_cast<char *>(pixel), 3);
        }
        outFile.write(reinterpret_cast<char *>(pad), padSize);
    }
}

inline double getDurationOf(std::function<void()> func) {
    auto t0 = std::chrono::high_resolution_clock::now();
    func();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> fs = t1 - t0;
    return std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(fs).count();
}


static UNUSED void printPerformanceCounts(InferenceEngine::InferenceEnginePluginPtr plugin, std::ostream &stream) {
    long long totalTime = 0;
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfomanceMap;
    // Get perfomance counts
    plugin->GetPerformanceCounts(perfomanceMap, nullptr);
    // Print perfomance counts
    stream << std::endl << "Perfomance counts:" << std::endl << std::endl;
    for (std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>::const_iterator it = perfomanceMap.begin();
         it != perfomanceMap.end(); ++it) {
        stream << std::setw(30) << std::left << it->first + ":";
        switch (it->second.status) {
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
        stream << std::setw(20) << std::left << "realTime: " + std::to_string(it->second.realTime_uSec);
        stream << std::setw(20) << std::left << " cpu: "  + std::to_string(it->second.cpu_uSec);
        stream << std::endl;

        if (it->second.realTime_uSec > 0) {
            totalTime += it->second.realTime_uSec;
        }
    }
    stream << std::setw(20) << std::left << "Total time: " + std::to_string(totalTime) << " microseconds" << std::endl;
}

/**
 * @brief This class represents an object that is found by an object detection net
 */
class DetectedObject {
public:
    int objectType;
    float xmin, xmax, ymin, ymax, prob;

    DetectedObject(int objectType, float xmin, float ymin, float xmax, float ymax, float prob)
        : objectType(objectType), xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), prob(prob) {
    }

    DetectedObject(const DetectedObject& other) = default;

    static float ioU(const DetectedObject& detectedObject1_, const DetectedObject& detectedObject2_) {
        // Add small space to eliminate empty squares
        float epsilon = 1e-3;

        DetectedObject detectedObject1(detectedObject1_.objectType,
                detectedObject1_.xmin - epsilon,
                detectedObject1_.ymin - epsilon,
                detectedObject1_.xmax + epsilon,
                detectedObject1_.ymax + epsilon, detectedObject1_.prob);
        DetectedObject detectedObject2(detectedObject2_.objectType,
                detectedObject2_.xmin - epsilon,
                detectedObject2_.ymin - epsilon,
                detectedObject2_.xmax + epsilon,
                detectedObject2_.ymax + epsilon, detectedObject2_.prob);

        if (detectedObject1.objectType != detectedObject2.objectType) {
            // objects are different, so the result is 0
            return 0.0f;
        }

        if (detectedObject1.xmax < detectedObject1.xmin) return 0.0;
        if (detectedObject1.ymax < detectedObject1.ymin) return 0.0;
        if (detectedObject2.xmax < detectedObject2.xmin) return 0.0;
        if (detectedObject2.ymax < detectedObject2.ymin) return 0.0;


        float xmin = (std::max)(detectedObject1.xmin, detectedObject2.xmin);
        float ymin = (std::max)(detectedObject1.ymin, detectedObject2.ymin);
        float xmax = (std::min)(detectedObject1.xmax, detectedObject2.xmax);
        float ymax = (std::min)(detectedObject1.ymax, detectedObject2.ymax);


        // intersection
        float intr;
        if ((xmax >= xmin) && (ymax >= ymin)) {
            intr = (xmax - xmin) * (ymax - ymin);
        } else {
            intr = 0.0f;
        }

        // union
        float square1 = (detectedObject1.xmax - detectedObject1.xmin) * (detectedObject1.ymax - detectedObject1.ymin);
        float square2 = (detectedObject2.xmax - detectedObject2.xmin) * (detectedObject2.ymax - detectedObject2.ymin);

        float unn = square1 + square2 - intr;

        return static_cast<float>((intr) / unn);
    }
};
