// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define _USE_MATH_DEFINES

#include "utils.hpp"

cv::Scalar getNetShape(const std::string& path) {
    InferenceEngine::Core ie;
    const auto network = ie.ReadNetwork(path);
    const auto layerName = network.getInputsInfo().begin()->first;
    const auto layerData = network.getInputsInfo().begin()->second;
    const auto layerDims = layerData->getTensorDesc().getDims();

    const int step = layerDims.size() == 5 ? 1 : 0;
    return cv::Scalar(double(layerDims[0 + step]),
                      double(layerDims[1 + step]),
                      double(layerDims[2 + step]),
                      double(layerDims[3 + step]));
}

// FIXME: cv::FileStorage can't open this .json files
void erase(std::string& str, const char symbol) {
    str.erase(std::remove(str.begin(), str.end(), symbol), str.end());
};

template<typename... Sargs>
void erase(std::string& str, const char symbol, Sargs... symbols) {
    erase(str, symbol);
    erase(str, symbols...);
};

std::vector<std::string> fill_labels(const std::string& dir) {
    std::ifstream fstream(dir);
    std::vector<std::string> labels;
    if (fstream.is_open()) {
        while(fstream) {
            std::string label;
            getline(fstream, label);
            erase(label, '"', ',', '[', ']');
            labels.push_back(label);
        }
        fstream.close();
        labels.erase(std::remove(labels.begin(), labels.end(), ""), labels.end());
    } else {
        throw std::logic_error("Gesture file doesn't open.");
    }
    return labels;
}
