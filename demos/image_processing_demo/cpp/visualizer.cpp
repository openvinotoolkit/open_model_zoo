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


Visualizer::Visualizer(const std::string& type) {
    if (type == "sr")
        winName = "Image Processing Demo - Super Resolution (press A for help)";
    else if (type == "deblur")
        winName = "Image Processing Demo - Deblurring (press A for help)";
}

cv::Size Visualizer::getSize() {
    return resolution;
}

void Visualizer::handleKey(int key) {
    key = std::tolower(key);
    if (key == 'a') {
        isHelpShown = !isHelpShown;
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

cv::Mat Visualizer::renderResultData(ImageResult result, cv::Size& newResolution) {
    if (!result.metaData) {
        throw std::invalid_argument("Renderer: metadata is null");
    }
    // Input image is stored inside metadata, as we put it there during submission stage
    inputImg = result.metaData->asRef<ImageMetaData>().img;
    if (inputImg.empty()) {
        throw std::invalid_argument("Renderer: image provided in metadata is empty");
    }

    if (!isResolutionSet) {
        setResolution(newResolution);
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

    if (isHelpShown) {
        int pad = 10;
        int margin = 40;
        int baseline = 0;
        int lineH = cv::getTextSize(helpMessage[0], cv::FONT_HERSHEY_COMPLEX_SMALL, 0.75, 1, &baseline).height + pad;
        for (size_t i = 0; i < 4; ++i) {
            cv::putText(img, helpMessage[i], cv::Point(pad, margin + baseline + (i + 1)*lineH),  cv::FONT_HERSHEY_COMPLEX_SMALL,
                        0.75, cv::Scalar(255, 0, 255));
        }
    }

    cv::imshow(winName, img);
}

void Visualizer::changeDisplayImg() {
    displayImg = resultImg.clone();
    if (mode == "orig") {
        inputImg(cv::Rect(0, 0, slider, inputImg.rows)).copyTo(displayImg(cv::Rect(0, 0, slider, resultImg.rows)));
        markImage(displayImg, {"O", "R"}, static_cast<float>(slider) / resolution.width);
        drawSweepLine(displayImg);
    } else if (mode == "result") {
        markImage(displayImg, {"R", ""}, 1);
    } else if (mode == "diff") {
        cv::Mat diffImg;
        cv::absdiff(inputImg, resultImg, diffImg);
        diffImg(cv::Rect(0, 0, slider, resultImg.rows)).copyTo(displayImg(cv::Rect(0, 0, slider, displayImg.rows)));
        markImage(displayImg, {"D", "R"}, static_cast<float>(slider) / resolution.width);
        drawSweepLine(displayImg);
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

void Visualizer::drawSweepLine(cv::Mat& image) {
    cv::line(image, cv::Point(slider, 0), cv::Point(slider, image.rows), cv::Scalar(0, 255, 0), 2);
}

void Visualizer::setResolution(cv::Size& newResolution) {
    resolution = newResolution;
    isResolutionSet = true;
    cv::setTrackbarMax(trackbarName, winName, resolution.width);
}

void Visualizer::addTrackbar() {
    if (!isTrackbarShown) {
        cv::createTrackbar(trackbarName, winName, &slider, resolution.width);
        cv::setTrackbarMin(trackbarName, winName, 1);
        isTrackbarShown = true;
    }
}

void Visualizer::disableTrackbar() {
    if (isTrackbarShown) {
        cv::destroyWindow(winName);
        isTrackbarShown = false;
        cv::namedWindow(winName);
        show();
    }
}
