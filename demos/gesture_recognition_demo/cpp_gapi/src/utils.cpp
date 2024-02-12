// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <algorithm>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>

#include <openvino/openvino.hpp>

#define _USE_MATH_DEFINES

cv::Scalar getNetShape(const std::string& path) {
    ov::Shape shape = ov::Core{}.read_model(path)->input().get_shape();
    const int step = shape.size() == 5 ? 1 : 0;
    return cv::Scalar(static_cast<double>(shape[0 + step]),
                      static_cast<double>(shape[1 + step]),
                      static_cast<double>(shape[2 + step]),
                      static_cast<double>(shape[3 + step]));
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
