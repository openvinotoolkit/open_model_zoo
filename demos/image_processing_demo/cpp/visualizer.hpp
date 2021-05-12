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
#include <pipelines/metadata.h>
#include <models/results.h>

namespace visualize {
    // names of window and trackbar
    std::string winName = "Image Processing Demo (press H for help)";
    std::string trackbarName = "Processing";

    // images info
    cv::Mat inputImg;
    cv::Mat resultImg;
    cv::Size maxSize = cv::Size(960, 540);

    // trackbar info
    std::string mode;
    const std::set<std::string> modes = {"", "orig", "diff"};
    int slider = 0;

    // help message
    bool helpShown = false;
    std::string helpMessage("Use mouse with LMB to paint\n"
                     "Use R to display the result\n"
                     "Use O to display the orig with result\n"
                     "Use D to display the diff with result\n"
                     "Esc or Q to quit\n");

    void init(const std::string& _mode, cv::Size resolution) {
        cv::namedWindow(winName, cv::WINDOW_AUTOSIZE);
        mode = _mode;
        maxSize = resolution;
        inputImg = cv::Mat(resolution, CV_32FC3, 0.);
        resultImg = cv::Mat(resolution, CV_32FC3, 0.);
        cv::createTrackbar(trackbarName, winName, &slider, maxSize.width);
        cv::setTrackbarMin(trackbarName, winName, 1);
    }

    void markImage(cv::Mat& image, const std::pair<std::string, std::string>& marks, float alpha){
        int pad = 25;
        std::pair<float, float> positions(static_cast<float>(image.cols) * alpha / 2.0f,
                                                 static_cast<float>(image.cols) * (1 + alpha) / 2.0f);
        cv::putText(image, marks.first, cv::Point(static_cast<int>(positions.first) - pad, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        cv::putText(image, marks.second, cv::Point(static_cast<int>(positions.second), 25),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
    }

    cv::Mat getDisplayImg() {
        cv::Mat displayImg;
        if (mode == "orig") {
            displayImg = inputImg.clone();
            resultImg(cv::Rect(0, 0, slider, inputImg.rows)).copyTo(displayImg(cv::Rect(0, 0, slider, resultImg.rows)));
            markImage(displayImg, {"R", "O"}, static_cast<float>(slider) / maxSize.width);
        } else if (mode == "") {
            displayImg = resultImg.clone();
            markImage(displayImg, {"R", ""}, 1);
        } else if (mode == "diff") {
            cv::absdiff(inputImg, resultImg, displayImg);
            resultImg(cv::Rect(0, 0, slider, resultImg.rows)).copyTo(displayImg(cv::Rect(0, 0, slider, displayImg.rows)));
            markImage(displayImg, {"R", "D"}, static_cast<float>(slider) / maxSize.width);
        }

        return displayImg;
    }

    cv::Mat renderResultData(ImageResult result) {
        if (!result.metaData) {
            throw std::invalid_argument("Renderer: metadata is null");
        }

        // Input image is stored inside metadata, as we put it there during submission stage
        inputImg = result.metaData->asRef<ImageMetaData>().img;
        if (inputImg.empty()) {
            throw std::invalid_argument("Renderer: image provided in metadata is empty");
        }
        cv::resize(result.resultImage, result.resultImage, maxSize);
        cv::resize(inputImg, inputImg, maxSize);

        if (inputImg.channels() != result.resultImage.channels())
            cv::cvtColor(result.resultImage, resultImg, cv::COLOR_GRAY2BGR);
        else
            resultImg = result.resultImage;

        cv::Mat displayImg = getDisplayImg();
        return displayImg;
    }

    void handleKey(int key) {
        key = std::tolower(key);
        if (key == 'a') {
            helpShown = 1 - helpShown;
        }
        if (key == 'o')
            mode = "orig";
        if (key == 'v')
            mode = "diff";
        if (key == 'r')
            mode = "";
    }

    void show(cv::Mat& img) {
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
        cv::imshow(visualize::winName, img);
    }
};
