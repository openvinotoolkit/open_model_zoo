// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <array>

#include <opencv2/core.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/infer.hpp>
#include <inference_engine.hpp>

namespace custom {
class NetsConfig {
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork tdNetwork;
    std::string         tdModelPath = "";
    std::vector<size_t> tdInputDims{};
    std::string         tdInputName = "";
    std::array<std::string,2> tdOutputNames{};

    InferenceEngine::CNNNetwork trNetwork;
    std::string trModelPath = "";
    std::string trOutputName = "";

    std::array<std::string,2> encoderOutputNames{};
    std::array<std::string,3> decoderInputNames{};
    std::array<std::string,2> decoderOutputNames{};
public:
    NetsConfig() = delete;
    NetsConfig(const std::string& tdModelPath_, const std::string& trModelPath_);

    void getTDinfo();
    void configureTD(const std::string& tdDevice,
                     const size_t tdNewInputWidth, const size_t tdNewInputHeight);
    void getTRinputInfo();
    void getTRoutputInfo(const std::string& trOutputBlobName, const bool trPadSymbolFirst,
                         const char kPadSymbol, const int decoderStartIndex,
                         const std::string& trSymbolsSet);
    void configureTR(const std::string& trDevice);
    void getTRcompositeInfo(const std::array<std::string,2>& encoderOutputNames_,
                            const std::array<std::string,3>& decoderInputNames_,
                            const std::array<std::string,2>& decoderOutputNames_,
                            const bool trPadSymbolFirst, const char kPadSymbol,
                            const std::string& trSymbolsSet, const std::string& decoderType);
    void configureTRcomposite(const std::string& trDevice);

    cv::Size tdInputSize{};
    size_t   tdInputChannels = 0;

    std::vector<size_t> trInputDims{};
    size_t              trInputChannels = 0;
    std::string decoderModelPath = "";
    std::vector<int> decoderHiddenInputDims{};
    std::vector<int> decoderFeaturesInputDims{};
    size_t decoderNumClasses = 0;
    size_t decoderEndToken = 0;

    std::string trAlphabet = "";

    cv::gapi::GNetPackage networks;
};
}
