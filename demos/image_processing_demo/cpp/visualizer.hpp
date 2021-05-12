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

class Visualizer {
private:
    // names of window and trackbar
    std::string winName = "Image Processing Demo (press A for help)";
    std::string trackbarName = "Processing";

    // images info
    cv::Mat inputImg;
    cv::Mat resultImg;
    cv::Mat displayImg;
    bool isSetResolution;
    cv::Size resolution = cv::Size(960, 540);

    // trackbar info
    std::string mode;
    bool trShown;
    int slider = 0;

    // help message
    bool helpShown = false;
    std::string helpMessage = "Use mouse with LMB to change slider position\n"
                              "Use R to display the result\n"
                              "Use O to display the orig with result\n"
                              "Use V to display the diff with result\n"
                              "Esc or Q to quit\n";
    void addTrackbar();
    void disableTrackbar();
    void setResolution(OutputTransform& transform);
    void markImage(cv::Mat& image, const std::pair<std::string, std::string>& marks, float alpha);
    void changeDisplayImg();
public:
    Visualizer();

    cv::Size getSize();

    // change display image for new input and result images
    cv::Mat renderResultData(ImageResult result, OutputTransform& transform);

    // show display image or specified value
    void show(cv::Mat img=cv::Mat());

    void handleKey(int key);
};
