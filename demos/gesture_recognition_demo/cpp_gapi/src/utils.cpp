// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <algorithm>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>

#include <cpp/ie_cnn_network.h>
#include <ie_core.hpp>
#include <ie_input_info.hpp>
#include <ie_layouts.h>

cv::Scalar getNetShape(const std::string& path) {
    const auto network = InferenceEngine::Core{}.ReadNetwork(path);
    const auto layerData = network.getInputsInfo().begin()->second;
    const auto layerDims = layerData->getTensorDesc().getDims();

    const int step = layerDims.size() == 5 ? 1 : 0;
    return cv::Scalar(static_cast<double>(layerDims[0 + step]),
                      static_cast<double>(layerDims[1 + step]),
                      static_cast<double>(layerDims[2 + step]),
                      static_cast<double>(layerDims[3 + step]));
}

void erase(std::string& str, const char symbol) {
    str.erase(std::remove(str.begin(), str.end(), symbol), str.end());
}

template <typename... Sargs>
void erase(std::string& str, const char symbol, Sargs... symbols) {
    erase(str, symbol);
    erase(str, symbols...);
}

std::vector<std::string> fill_labels(const std::string& dir) {
    std::ifstream fstream(dir);
    std::vector<std::string> labels;
    if (fstream.is_open()) {
        while (fstream) {
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
