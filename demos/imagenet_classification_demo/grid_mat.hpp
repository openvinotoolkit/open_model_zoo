// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <set>
#include <string>
#include <vector>
#include <queue>

#include <monitors/presenter.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>

enum class PredictionResult { Correct,
                              Incorrect,
                              Unknown };
using ImageInfoList = std::list<std::tuple<cv::Mat, std::string, PredictionResult>>;

class GridMat {
public:
    cv::Mat outImg;

    explicit GridMat(Presenter& presenter,
                     const cv::Size maxDisp = cv::Size{1920, 1080},
                     const cv::Size aspectRatio = cv::Size{16, 9},
                     double targetFPS = 60
                     ):
                     currSourceId{0} {
        targetFPS = std::max(targetFPS, static_cast<double>(FLAGS_b));
        cv::Size size = cv::Size(static_cast<int>(
                                    std::round(sqrt(1. * targetFPS * aspectRatio.width / aspectRatio.height))),
                        static_cast<int>(std::round(sqrt(1. * targetFPS * aspectRatio.height / aspectRatio.width))));
        int minCellSize = std::min(maxDisp.width / size.width, maxDisp.height / size.height);
        cellSize = cv::Size(minCellSize, minCellSize);

        for (int i = 0; i < size.width; i++) {
            for (int j = 0; j < size.height; j++) {
                points.emplace_back(cellSize.width * i, presenter.graphSize.height + cellSize.height * j);
            }
        }

        outImg.create((cellSize.height * size.height) + presenter.graphSize.height,
                       cellSize.width * size.width, CV_8UC3);
        outImg.setTo(0);

        fontType = cv::FONT_HERSHEY_PLAIN;
        fontScale = 1.5;
        thickness = 2;
        textSize = cv::getTextSize("", fontType, fontScale, thickness, &baseline);
        accuracyMessageSize = cv::getTextSize("Accuracy (top 0): 0.000", fontType, fontScale, thickness, &baseline);
        testMessage = "Testing, please wait...";
        testMessageSize = cv::getTextSize(testMessage, fontType, fontScale, thickness, &baseline);
    }

    void textUpdate(double avgFPS, double avgLatency, double accuracy,
                    bool isFpsTest, bool showAccuracy,
                    Presenter& presenter) {
        rectangle(outImg,
                  {0, 0}, {outImg.cols, presenter.graphSize.height},
                  cv::Scalar(0, 0, 0), cv::FILLED);

        presenter.drawGraphs(outImg);
        
        cv::Scalar textColor = cv::Scalar(255, 255, 255);
        int textPadding = 10;

        cv::putText(outImg,
                    cv::format("FPS: %0.01f", avgFPS),
                    cv::Point(textPadding, textSize.height + textPadding),
                    fontType, fontScale, textColor, thickness);
        cv::putText(outImg,
                    cv::format("Latency: %dms", static_cast<int>(avgLatency * 1000)),
                    cv::Point(textPadding, (textSize.height + textPadding) * 2),
                    fontType, fontScale, textColor, thickness);
        
        if (showAccuracy) {
            cv::putText(outImg,
                        cv::format("Accuracy (top %d): %.3f", FLAGS_nt, accuracy),
                        cv::Point(outImg.cols - accuracyMessageSize.width - textPadding, textSize.height + textPadding),
                        fontType, fontScale, textColor, thickness);
        }

        if (isFpsTest) {
            cv::putText(outImg,
                        testMessage,
                        cv::Point(outImg.cols - testMessageSize.width - textPadding,
                                  (textSize.height + textPadding) * 2),
                        fontType, fontScale, cv::Scalar(50, 50, 255), thickness);
        }
    }

    void updateMat(const ImageInfoList& imageInfos) {
        size_t prevSourceId = (currSourceId + points.size() - prevImgs.size() % points.size()) % points.size();
        while (!prevImgs.empty()) {
            if (prevSourceId >= points.size()) {
                prevSourceId -= points.size();
            }

            prevImgs.front().copyTo(outImg(cv::Rect(points[prevSourceId], cellSize)));
            prevImgs.pop();
            prevSourceId++;
        }

        for (const auto & imageInfo : imageInfos) {
            cv::Mat frame = std::get<0>(imageInfo);
            std::string predictedLabel = std::get<1>(imageInfo);
            PredictionResult predictionResult = std::get<2>(imageInfo);

            cv::Scalar textColor;
            switch (predictionResult) {
                case PredictionResult::Correct:
                    textColor = cv::Scalar(75, 255, 75); break;     // green
                case PredictionResult::Incorrect:
                    textColor = cv::Scalar(50, 50, 255); break;     // red
                case PredictionResult::Unknown:
                    textColor = cv::Scalar(75, 255, 255); break;    // yellow
                default:
                    throw std::runtime_error("Undefined type of prediction result");
            }

            int labelThickness = cellSize.width / 20;
            cv::Size labelTextSize = cv::getTextSize(predictedLabel, fontType, 1, 2, &baseline);
            double labelFontScale = static_cast<double>(cellSize.width - 2*labelThickness) / labelTextSize.width;
            cv::resize(frame, frame, cellSize);
            cv::putText(frame,
                        predictedLabel,
                        cv::Point(labelThickness, cellSize.height - labelThickness - labelTextSize.height),
                        fontType, labelFontScale, textColor, 2);
            
            prevImgs.push(frame);

            cv::Mat cell = outImg(cv::Rect(points[currSourceId], cellSize));                                         
            frame.copyTo(cell);                                                                                      
            cv::rectangle(cell, {0, 0}, {frame.cols, frame.rows}, {255, 50, 50}, labelThickness);
            
            if (currSourceId == points.size() - 1) {
                currSourceId = 0;
            } else {
                currSourceId++;
            }
        }
    }

private:
    std::queue<cv::Mat> prevImgs;
    cv::Size cellSize;
    size_t currSourceId;
    std::vector<cv::Point> points;
    int fontType;
    double fontScale;
    int thickness;
    int baseline;
    cv::Size textSize;
    cv::Size accuracyMessageSize;
    std::string testMessage;
    cv::Size testMessageSize;
};
