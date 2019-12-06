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
    explicit GridMat(int inputImgsCount,
                     const cv::Size maxDisp = cv::Size{1920, 1080},
                     const cv::Size aspectRatio = cv::Size{16, 9},
                     int scale = 2
                     ):
                     size{static_cast<int>(std::ceil(1. * scale * scale / aspectRatio.height)),
                          static_cast<int>(std::ceil(1. * scale * scale / aspectRatio.width))},
                     currSourceID{0},
                     rectangleHeight{30} {
        if (size.width % inputImgsCount == 0) { // fix 'static GridMat' problem
            size.width++;
        }
        else if (size.height % inputImgsCount == 0) {
            size.height++;
        }

        int minCellSize = std::min(maxDisp.width / size.width, maxDisp.height / size.height);
        cellSize = cv::Size(minCellSize, minCellSize);

        for (int i = 0; i < size.width; i++) {
            for (int j = 0; j < size.height; j++) {
                cv::Point p;
                p.x = cellSize.width * i;
                p.y = rectangleHeight + (cellSize.height * j);
                points.push_back(p);
            }
        }

        outImg.create((cellSize.height * size.height) + rectangleHeight, cellSize.width * size.width, CV_8UC3);
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

    void textUpdate(double overFPS, bool isFpsTest){
        // Draw a rectangle
        cv::Point p1 = cv::Point(0, 0);
        cv::Point p2 = cv::Point(outImg.cols, rectangleHeight);
        rectangle(outImg, p1, p2, cv::Scalar(0, 0, 0), cv::FILLED);
        
        // Show FPS
        double fontScale = 2;
        int thickness = 2;

        if (!isFpsTest) {
            cv::putText(outImg,
                        cv::format("Overall FPS: %0.01f", overFPS),
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
            cv::Mat frame = updateList.front();
            updateList.pop_front();

            if (updateList.empty()) {
                if (prevImgs.size() >= FLAGS_b && points.size() > 1) {
                    int prevSourceID = currSourceID - FLAGS_b;
                    if (prevSourceID < 0) {
                        prevSourceID += points.size();
                    }
                    cv::resize(prevImgs.front(),
                               outImg(cv::Rect(points[prevSourceID], cellSize)),
                               cellSize);
                    prevImgs.pop();
                }

                prevImgs.push(frame);
                
                int thickness =  static_cast<int>(10. * frame.cols / cellSize.width);
                copyMakeBorder(frame,
                               frame,
                               thickness, thickness, thickness, thickness,
                               cv::BORDER_CONSTANT,
                               cv::Scalar(0, 255, 0));
                
                cv::resize(frame,
                           outImg(cv::Rect(points[currSourceID], cellSize)),
                           cellSize);
            
                if (currSourceID == points.size() - 1)
                    currSourceID = 0;
                else
                    currSourceID++;
            }
        }

        return outImg;
    }

private:
    cv::Mat outImg;
    std::queue<cv::Mat> prevImgs;
    std::list<cv::Mat> updateList;
    cv::Size size;
    cv::Size cellSize;
    size_t currSourceID;
    std::vector<cv::Point> points;
    unsigned rectangleHeight;
};
