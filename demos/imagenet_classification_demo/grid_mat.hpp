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
                     double targetFPS = 60
                     ):
                     size{std::max(static_cast<int>(std::ceil(targetFPS / aspectRatio.height)),
                                   static_cast<int>(std::ceil(
                                       1. * FLAGS_b / (aspectRatio.width + aspectRatio.height) * aspectRatio.width))
                                    ),
                          std::max(static_cast<int>(std::ceil(targetFPS / aspectRatio.width)),
                                   static_cast<int>(std::ceil(
                                       1. * FLAGS_b / (aspectRatio.width + aspectRatio.height) * aspectRatio.height))
                                    )},
                     currSourceID{0},
                     rectangleHeight{30} {
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

    void listUpdate(std::list<std::tuple<cv::Mat, std::string, bool>>& imageInfos) {
        if (!imageInfos.empty()) {
            updateList.splice(updateList.end(), imageInfos);
        }
    }

    void textUpdate(double avgFPS, double avgLatency, bool isFpsTest) {
        // Draw a rectangle
        cv::Point p1 = cv::Point(0, 0);
        cv::Point p2 = cv::Point(outImg.cols, rectangleHeight);
        rectangle(outImg, p1, p2, cv::Scalar(0, 0, 0), cv::FILLED);
        
        // Show FPS
        double fontScale = 2;
        int thickness = 2;

        if (!isFpsTest) {
            cv::putText(outImg,
                        cv::format("FPS: %0.01f     Latency: %dms", avgFPS, static_cast<int>(avgLatency*1000)),
                        cv::Point(5, rectangleHeight - 5),
                        cv::FONT_HERSHEY_PLAIN, fontScale, cv::Scalar(0, 255, 0), thickness);
        } else {
            cv::putText(outImg,
                        cv::format("FPS: %0.01f     Latency: %dms     Testing, please wait...",
                                   avgFPS,
                                   static_cast<int>(avgLatency*1000)),
                        cv::Point(5, rectangleHeight - 5),
                        cv::FONT_HERSHEY_PLAIN, fontScale, cv::Scalar(0, 0, 255), thickness);
        }
    }

    cv::Mat getMat() {
        while (!updateList.empty()) {
            cv::Mat frame = std::get<0>(updateList.front());
            std::string predictedLabel = std::get<1>(updateList.front());
            cv::Scalar textColor;
            if (std::get<2>(updateList.front())) { // if prediction is right
                textColor = cv::Scalar(0, 255, 0); // green color
            }
            else {
                textColor = cv::Scalar(0, 0, 255); // red color
            }
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
                cv::Mat tmpFrame;
                frame.copyTo(tmpFrame);
                cv::rectangle(tmpFrame, cv::Point(), cv::Point(frame.cols, frame.rows), cv::Scalar(255, 0, 0), thickness);
                
                cv::resize(tmpFrame,
                           outImg(cv::Rect(points[currSourceID], cellSize)),
                           cellSize);
                
                int fontFace = cv::FONT_HERSHEY_PLAIN;
                double fontScale = 1;
                thickness = 2;
                int baseline = 0;
                cv::Size textSize = cv::getTextSize(predictedLabel, fontFace, fontScale, thickness, &baseline);
                fontScale = static_cast<double>(cellSize.width) / textSize.width;

                cv::putText(outImg,
                        predictedLabel,
                        cv::Point(points[currSourceID].x, points[currSourceID].y + cellSize.height),
                        cv::FONT_HERSHEY_PLAIN, fontScale, textColor, thickness);
            
                if (currSourceID == points.size() - 1) {
                    currSourceID = 0;
                } else {
                    currSourceID++;
                }
            }
        }

        return outImg;
    }

private:
    cv::Mat outImg;
    std::queue<cv::Mat> prevImgs;
    std::list<std::tuple<cv::Mat, std::string, bool>> updateList;
    cv::Size size;
    cv::Size cellSize;
    size_t currSourceID;
    std::vector<cv::Point> points;
    unsigned rectangleHeight;
};
