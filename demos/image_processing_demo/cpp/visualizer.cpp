/*
// Copyright (C) 2021 Intel Corporation
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

#include <iostream>
#include <set>
#include <string>
#include "visualizer.hpp"

Visualizer::Visualizer() {
    cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
    mode = "result";
    inputImg = cv::Mat(resolution, CV_32FC3, 0.);
    resultImg = cv::Mat(resolution, CV_32FC3, 0.);
    displayImg = cv::Mat(resolution, CV_32FC3, 0.);
    trShown = false;
    isSetResolution = false;
}

cv::Size Visualizer::getSize() {
    return resolution;
}

void Visualizer::handleKey(int key) {
    key = std::tolower(key);
    if (key == 'a') {
        helpShown = 1 - helpShown;
    }
    if (key == 'o') {
        mode = "orig";
        addTrackbar();
    }
    if (key == 'v') {
        mode = "diff";
        addTrackbar();
    }
    if (key == 'r') {
        mode = "result";
        disableTrackbar();
    }
}

cv::Mat Visualizer::renderResultData(ImageResult result, OutputTransform& transform) {
    if (!result.metaData) {
        throw std::invalid_argument("Renderer: metadata is null");
    }
    // Input image is stored inside metadata, as we put it there during submission stage
    inputImg = result.metaData->asRef<ImageMetaData>().img;
    if (inputImg.empty()) {
        throw std::invalid_argument("Renderer: image provided in metadata is empty");
    }

    if (!isSetResolution) {
        transform.resize(result.resultImage);
        setResolution(transform);
    }

    cv::resize(result.resultImage, result.resultImage, resolution);
    cv::resize(inputImg, inputImg, resolution);

    if (inputImg.channels() != result.resultImage.channels()){
        cv::cvtColor(result.resultImage, resultImg, cv::COLOR_GRAY2BGR);
    }
    else
        resultImg = result.resultImage;
    changeDisplayImg();
    return displayImg;
}

void Visualizer::show(cv::Mat img) {
    if (img.empty()) {
        changeDisplayImg();
        img = displayImg;
    }

    if (helpShown) {
        float pad = 10;
        float margin = 40;
        std::vector<std::string> lines;
        std::string line;
        std::istringstream textStream(helpMessage);
        while (std::getline(textStream, line, '\n'))
           lines.push_back(line);
        int baseline = 0;
        int lineH = cv::getTextSize(lines[0], cv::FONT_HERSHEY_COMPLEX, 0.75, 1, &baseline).height + pad;
        for (size_t i = 0; i < lines.size(); ++i) {
            cv::putText(img, lines[i], cv::Point(pad, margin + baseline + (i + 1)*lineH),  cv::FONT_HERSHEY_COMPLEX,
                        0.75, cv::Scalar(255, 0, 255));
        }
    }
    cv::imshow(winName, img);
}

void Visualizer::changeDisplayImg() {
    if (mode == "orig") {
        displayImg = inputImg.clone();
        resultImg(cv::Rect(0, 0, slider, inputImg.rows)).copyTo(displayImg(cv::Rect(0, 0, slider, resultImg.rows)));
        markImage(displayImg, {"R", "O"}, static_cast<float>(slider) / resolution.width);
    } else if (mode == "result") {
        displayImg = resultImg.clone();
        markImage(displayImg, {"R", ""}, 1);
    } else if (mode == "diff") {
        cv::absdiff(inputImg, resultImg, displayImg);
        resultImg(cv::Rect(0, 0, slider, resultImg.rows)).copyTo(displayImg(cv::Rect(0, 0, slider, displayImg.rows)));
        markImage(displayImg, {"R", "D"}, static_cast<float>(slider) / resolution.width);
    }
}

void Visualizer::markImage(cv::Mat& image, const std::pair<std::string, std::string>& marks, float alpha) {
    int pad = 25;
    std::pair<float, float> positions(static_cast<float>(image.cols) * alpha / 2.0f,
                                             static_cast<float>(image.cols) * (1 + alpha) / 2.0f);
    cv::putText(image, marks.first, cv::Point(static_cast<int>(positions.first) - pad, 25),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
    cv::putText(image, marks.second, cv::Point(static_cast<int>(positions.second), 25),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
}

void Visualizer::setResolution(OutputTransform& transform) {
    resolution = transform.computeResolution();
    isSetResolution = true;
    cv::setTrackbarMax(trackbarName, winName, resolution.width);
}

void Visualizer::addTrackbar() {
    if (!trShown) {
        cv::createTrackbar(trackbarName, winName, &slider, resolution.width);
        cv::setTrackbarMin(trackbarName, winName, 1);
        trShown = true;
    }
}

void Visualizer::disableTrackbar() {
    if (trShown) {
        cv::destroyWindow(winName);
        trShown = false;
        cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
        show();
    }
}
