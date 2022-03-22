// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <opencv2/gapi/infer/ie.hpp>

#include "utils/common.hpp"

#include "custom_nets.hpp"
#include "nets_configuration.hpp"

custom::NetsConfig::NetsConfig(const std::string& tdModelPath_, const std::string& trModelPath_)
    : tdModelPath(tdModelPath_), trModelPath(trModelPath_) {
    if ("" != tdModelPath_) {
        tdNetwork = ie.ReadNetwork(tdModelPath_);
    }
    if ("" != trModelPath) {
        trNetwork = ie.ReadNetwork(trModelPath);
    }
}

template<size_t n>
void checkIONames(const std::vector<std::string>& layers, const std::array<std::string,n>& names) {
    for (const auto& name : names) {
        if (std::find(layers.begin(), layers.end(), name) == layers.end()) {
            throw std::runtime_error("Name '" + name +
                                     "' does not exist in the network");
        }
    }
}
void checkCompositeNetNames(const std::vector<std::string>&  encOutputLayers,
                            const std::array<std::string,2>& encOutputNames,
                            const std::vector<std::string>&  decInputLayers,
                            const std::array<std::string,3>& decInputNames,
                            const std::vector<std::string>&  decOutputLayers,
                            const std::array<std::string,2>& decOutputNames) {
    checkIONames(encOutputLayers, encOutputNames);
    checkIONames(decInputLayers, decInputNames);
    checkIONames(decOutputLayers, decOutputNames);
}

void custom::NetsConfig::getTDinfo() {
    // Getting text detection network input info
    auto inputInfo = tdNetwork.getInputsInfo();
    if (1 != inputInfo.size()) {
        throw std::runtime_error("The text detection network should have "
                                    "only one input");
    }
    tdInputName = inputInfo.begin()->first;
    tdInputDims = inputInfo.begin()->second->getInputData()->getTensorDesc().getDims();
    if (4 != tdInputDims.size()) {
        throw std::runtime_error("The text detection network should have "
                                    "4-dimensional input");
    }
    tdInputSize = cv::Size(tdInputDims[3], tdInputDims[2]);
    tdInputChannels = tdInputDims[1];
    // Getting text detection network output names
    const size_t tdLinkLayerChannels = 16;
    const size_t tdSegmLayerChannels = 2;
    const size_t tdHorizBoxesLayerChannels  = 5;
    const size_t tdHorizLabelsLayerChannels = 0;
    auto outputInfo = tdNetwork.getOutputsInfo();
    for (const auto& pair : outputInfo) {
        switch (pair.second->getTensorDesc().getDims()[1]) {
        case tdLinkLayerChannels:
            tdOutputNames[0] = pair.first;
            break;
        case tdSegmLayerChannels:
            tdOutputNames[1] = pair.first;
            break;
        case tdHorizBoxesLayerChannels:
            tdOutputNames[0] = pair.first;
            break;
        case tdHorizLabelsLayerChannels:
            tdOutputNames[1] = pair.first;
            break;
        }
    }
    if (tdOutputNames[0].empty() || tdOutputNames[1].empty()) {
        throw std::runtime_error("Failed to determine text detection output layers' names");
    }
}

void custom::NetsConfig::configureTD(const std::string& tdDevice,
                                     const size_t tdNewInputWidth, const size_t tdNewInputHeight) {
    const cv::Size tdNewInputSize = cv::Size(tdNewInputWidth, tdNewInputHeight);
    const bool tdReshape          = cv::Size() != tdNewInputSize;
    auto tdNet = cv::gapi::ie::Params<nets::TextDetection> {
        tdModelPath, fileNameNoExt(tdModelPath) + ".bin", tdDevice
    }.cfgOutputLayers(tdOutputNames);
    if (tdReshape) {
        tdInputDims[2] = tdNewInputHeight;
        tdInputDims[3] = tdNewInputWidth;
        tdNet.cfgInputReshape(tdInputName, tdInputDims);
        tdInputSize = tdNewInputSize;
    }
    networks += cv::gapi::networks(tdNet);
}

void custom::NetsConfig::getTRinputInfo() {
    // Getting text recognition network input info
    auto inputInfo = trNetwork.getInputsInfo();
    if (1 != inputInfo.size()) {
        throw std::runtime_error("The text recognition network should have "
                                    "only one input");
    }
    trInputDims = inputInfo.begin()->second->getInputData()->
                                getTensorDesc().getDims();
    if (4 != trInputDims.size()) {
        throw std::runtime_error("The text recognition network should have "
                                    "4-dimensional input");
    }
    trInputChannels = trInputDims[1];
}

void custom::NetsConfig::getTRoutputInfo(const std::string& trOutputBlobName,
                                         const bool trPadSymbolFirst,
                                         const char kPadSymbol,
                                         const int decoderStartIndex,
                                         const std::string& trSymbolsSet) {
    // Getting text recognition network output info
    auto outputInfo = trNetwork.getOutputsInfo();
    if ("" != trOutputBlobName) {
        for (const auto& pair : outputInfo) {
            if (pair.first == trOutputBlobName) {
                trOutputName = trOutputBlobName;
                break;
            }
        }
        if (trOutputName.empty()) {
            throw std::runtime_error("The text recognition model does not have "
                                        " output " + trOutputBlobName);
        }
    } else {
        trOutputName = outputInfo.begin()->first;
    }
    trAlphabet = trPadSymbolFirst
        ? std::string(decoderStartIndex + 1, kPadSymbol) + trSymbolsSet
        : trSymbolsSet + kPadSymbol;
}

void custom::NetsConfig::configureTR(const std::string& trDevice) {
    auto trNet = cv::gapi::ie::Params<nets::TextRecognition> {
        trModelPath, fileNameNoExt(trModelPath) + ".bin", trDevice
    }.cfgOutputLayers({trOutputName});
    networks += cv::gapi::networks(trNet);
}

template<class Map> void getKeys(const Map& map, std::vector<std::string>& vec) {
    std::transform(map.begin(), map.end(), std::back_inserter(vec),
        [](const typename Map::value_type& pair) { return pair.first; });
}

void getIntDims(const std::vector<size_t>& sizeVector, std::vector<int>& dims) {
    std::transform(sizeVector.begin(), sizeVector.end(), std::back_inserter(dims),
        [](const size_t dim) { return dim; });
}

void custom::NetsConfig::getTRcompositeInfo(const std::array<std::string,2>& encoderOutputNames_,
                                            const std::array<std::string,3>& decoderInputNames_,
                                            const std::array<std::string,2>& decoderOutputNames_,
                                            const bool trPadSymbolFirst,
                                            const char kPadSymbol,
                                            const std::string& trSymbolsSet,
                                            const std::string& decoderType) {
    // This demo covers a certain composite `text-recognition-0015/0016` topology;
    // in case of different network this might need to be changed or generalized
    encoderOutputNames = encoderOutputNames_;
    decoderInputNames = decoderInputNames_;
    decoderOutputNames = decoderOutputNames_;
    // Text Recognition Decoder network
    decoderModelPath = trModelPath;
    while (std::string::npos != decoderModelPath.find("encoder")) {
        decoderModelPath = decoderModelPath.replace(
            decoderModelPath.find("encoder"), 7, "decoder");
    }
    auto decNetwork = ie.ReadNetwork(decoderModelPath);
    auto decInputInfo = decNetwork.getInputsInfo();

    // Checking names legitimacy
    std::vector<std::string> encoderOutputLayers {};
    std::vector<std::string> decoderInputLayers  {};
    std::vector<std::string> decoderOutputLayers {};
    getKeys(trNetwork.getOutputsInfo(),  encoderOutputLayers);
    getKeys(decInputInfo,                decoderInputLayers);
    getKeys(decNetwork.getOutputsInfo(), decoderOutputLayers);
    checkCompositeNetNames(encoderOutputLayers, encoderOutputNames,
                           decoderInputLayers,  decoderInputNames,
                           decoderOutputLayers, decoderOutputNames);

    decoderHiddenInputDims.clear();
    getIntDims(decInputInfo[decoderInputNames[1]]->getInputData()->getTensorDesc().getDims(),
               decoderHiddenInputDims);
    decoderFeaturesInputDims.clear();
    getIntDims(decInputInfo[decoderInputNames[2]]->getInputData()->getTensorDesc().getDims(),
               decoderFeaturesInputDims);

    trAlphabet = std::string(3, kPadSymbol) + trSymbolsSet;
    decoderNumClasses = trAlphabet.length();
    decoderEndToken = trAlphabet.find(kPadSymbol, 2);
    if (!trPadSymbolFirst) {
        throw std::logic_error("Flag '-tr_pt_first' was not set. "
                                "Set the flag if you want to use composite model");
    }
    if ("simple" != decoderType) {
        throw std::logic_error("Wrong decoder. "
                                "Use --dt simple for composite model.");
    }
}

void custom::NetsConfig::configureTRcomposite(const std::string& trDevice) {
    static auto trEncNet = cv::gapi::ie::Params<nets::TextRecognitionEncoding> {
        trModelPath, fileNameNoExt(trModelPath) + ".bin", trDevice
    }.cfgOutputLayers({encoderOutputNames});
    networks += cv::gapi::networks(trEncNet);

    static auto trDecNet = cv::gapi::ie::Params<nets::TextRecognitionDecoding> {
        decoderModelPath, fileNameNoExt(decoderModelPath) + ".bin", trDevice
    }.cfgInputLayers({decoderInputNames}).cfgOutputLayers({decoderOutputNames});
    networks += cv::gapi::networks(trDecNet);
}
