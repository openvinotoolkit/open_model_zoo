/*
// Copyright (C) 2021-2024 Intel Corporation
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

#include <string>
#include <utility>

#include <opencv2/core.hpp>

struct ImageResult;

class Visualizer {
private:
    // names of window and trackbar
    std::string winName = "Image Processing Demo (press A for help)";
    std::string trackbarName = "Orig/Diff | Res";

    // images info
    cv::Size resolution = cv::Size(1000, 600);
    bool isResolutionSet = false;
    cv::Mat inputImg = cv::Mat(resolution, CV_32FC3, 0.);
    cv::Mat resultImg = cv::Mat(resolution, CV_32FC3, 0.);
    cv::Mat displayImg = cv::Mat(resolution, CV_32FC3, 0.);

    // trackbar info
    std::string mode = "result";
    bool isTrackbarShown = false;
    int slider = 1;

    // help message
    bool isHelpShown = false;
    std::string helpMessage[4] = {"Use R to display the result",
                                  "Use O to display the orig with result",
                                  "Use V to display the diff with result",
                                  "Esc or Q to quit"};
    void addTrackbar();
    void disableTrackbar();
    void setResolution(cv::Size& newResolution);
    void markImage(cv::Mat& image, const std::pair<std::string, std::string>& marks, float alpha);
    void drawSweepLine(cv::Mat& image);
    void changeDisplayImg();

public:
    Visualizer(const std::string& type = "");
    cv::Size getSize();

    // change display image for new input and result images
    cv::Mat renderResultData(ImageResult result, cv::Size& newResolution);

    // show display image or specified value
    void show(cv::Mat img = cv::Mat());

    void handleKey(int key);
};
