// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <set>
#include <string>
#include <vector>
#include <queue>

#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>

class GridMat {
public:
    explicit GridMat(const cv::Size maxDisp = cv::Size{1920, 1080}, int scale = 2): currSourceID{0} {
        cv::Size aspectRatio = cv::Size{16, 9};
        size = cv::Size{static_cast<int>(std::ceil(scale * scale / double(aspectRatio.height))),
                        static_cast<int>(std::ceil(scale * scale / double(aspectRatio.width)))};
        cellSize = std::min(maxDisp.width / size.width, maxDisp.height / size.height);
        rectangleHeight = 30;

        for (int i = 0; i < size.width; i++) {
            for (int j = 0; j < size.height; j++) {
                cv::Point p;
                p.x = cellSize * i;
                p.y = rectangleHeight + (cellSize * j);
                points.push_back(p);
            }
        }

        outImg.create((cellSize * size.height) + rectangleHeight, cellSize * size.width, CV_8UC3);
        outImg.setTo(0);
    }

    cv::Size getSize() {
        return size;
    }

    void listUpdate(std::list<cv::Mat>& frames) {
        if (!frames.empty()) {
            updateList.splice(updateList.end(), frames);
        }
    }

    void textUpdate(double overSPF, bool isFpsTest){
        // Draw a rectangle
        cv::Point p1 = cv::Point(0, 0);
        cv::Point p2 = cv::Point(outImg.cols, rectangleHeight);
         
        rectangle(outImg, p1, p2, cv::Scalar(0, 0, 0), cv::FILLED);
        
        // Show FPS
        double fontScale = 2;
        int thickness = 2;

        if (!isFpsTest) {
            cv::putText(outImg,
                        cv::format("Overall FPS: %0.01f", 1./overSPF),
                        cv::Point(5, rectangleHeight - 5),
                        cv::FONT_HERSHEY_PLAIN, fontScale, cv::Scalar(0, 255, 0), thickness);
        }
        else {
            cv::putText(outImg,
                        "FPS test, please wait...",
                        cv::Point(5, rectangleHeight - 5),
                        cv::FONT_HERSHEY_PLAIN, fontScale, cv::Scalar(0, 0, 255), thickness);
        }
    }

    cv::Mat getMat() {
        while (!updateList.empty()) {
            cv::Mat cell = outImg(cv::Rect(points[currSourceID], cv::Size{cellSize, cellSize}));
            cv::Mat frame = updateList.front();
            updateList.pop_front();

            cv::resize(frame, cell, cv::Size{cellSize, cellSize});
            
            if (currSourceID == points.size() - 1)
                currSourceID = 0;
            else
                currSourceID++;
        }

        return outImg;
    }

private:
    cv::Mat outImg;
    std::list<cv::Mat> updateList;
    cv::Size size;
    int cellSize;
    size_t currSourceID;
    std::vector<cv::Point> points;
    size_t rectangleHeight;
};
