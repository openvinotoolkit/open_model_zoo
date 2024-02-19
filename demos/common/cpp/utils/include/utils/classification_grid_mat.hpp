// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <monitors/presenter.h>
#include <utils/ocv_common.hpp>

#include "utils/performance_metrics.hpp"

enum class PredictionResult { Correct, Incorrect, Unknown };

class ClassificationGridMat {
public:
    cv::Mat outImg;

    explicit ClassificationGridMat(Presenter& presenter,
                     const cv::Size maxDisp = cv::Size{1920, 1080},
                     const cv::Size aspectRatio = cv::Size{16, 9},
                     double targetFPS = 60)
        : currSourceId{0} {
        cv::Size size(static_cast<int>(std::round(sqrt(1. * targetFPS * aspectRatio.width / aspectRatio.height))),
                      static_cast<int>(std::round(sqrt(1. * targetFPS * aspectRatio.height / aspectRatio.width))));
        if (size.width == 0 || size.height == 0) {
            size = {1, 1};  // set minimum possible grid size
        }
        int minCellSize = std::min(maxDisp.width / size.width, maxDisp.height / size.height);
        cellSize = cv::Size(minCellSize, minCellSize);

        for (int i = 0; i < size.height; i++) {
            for (int j = 0; j < size.width; j++) {
                points.emplace_back(cellSize.width * j, presenter.graphSize.height + cellSize.height * i);
            }
        }

        outImg.create((cellSize.height * size.height) + presenter.graphSize.height,
                      cellSize.width * size.width,
                      CV_8UC3);
        outImg.setTo(0);

        textSize = cv::getTextSize("", fontType, fontScale, thickness, &baseline);
        accuracyMessageSize = cv::getTextSize("Accuracy (top 0): 0.000", fontType, fontScale, thickness, &baseline);
        testMessageSize = cv::getTextSize(ClassificationGridMat::testMessage, fontType, fontScale, thickness, &baseline);
    }

    void textUpdate(PerformanceMetrics& metrics,
                    PerformanceMetrics::TimePoint lastRequestStartTime,
                    double accuracy,
                    unsigned int nTop,
                    bool isFpsTest,
                    bool showAccuracy,
                    Presenter& presenter) {
        rectangle(outImg, {0, 0}, {outImg.cols, presenter.graphSize.height}, cv::Scalar(0, 0, 0), cv::FILLED);

        presenter.drawGraphs(outImg);

        metrics.update(lastRequestStartTime,
                       outImg,
                       cv::Point(textPadding, textSize.height + textPadding),
                       fontType,
                       fontScale,
                       cv::Scalar(255, 100, 100),
                       thickness);

        if (showAccuracy) {
            cv::putText(outImg,
                        cv::format("Accuracy (top %d): %.3f", nTop, accuracy),
                        cv::Point(outImg.cols - accuracyMessageSize.width - textPadding, textSize.height + textPadding),
                        fontType,
                        fontScale,
                        cv::Scalar(255, 255, 255),
                        thickness);
        }

        if (isFpsTest) {
            cv::putText(
                outImg,
                ClassificationGridMat::testMessage,
                cv::Point(outImg.cols - testMessageSize.width - textPadding, (textSize.height + textPadding) * 2),
                fontType,
                fontScale,
                cv::Scalar(50, 50, 255),
                thickness);
        }
    }

    void updateMat(const cv::Mat& mat, const std::string& label, PredictionResult predictionResul) {
        if (!prevImg.empty()) {
            size_t prevSourceId = currSourceId - 1;
            prevSourceId = std::min(prevSourceId, points.size() - 1);
            prevImg.copyTo(outImg(cv::Rect(points[prevSourceId], cellSize)));
        }
        cv::Scalar textColor;
        switch (predictionResul) {
            case PredictionResult::Correct:
                textColor = cv::Scalar(75, 255, 75);  // green
                break;
            case PredictionResult::Incorrect:
                textColor = cv::Scalar(50, 50, 255);  // red
                break;
            case PredictionResult::Unknown:
                textColor = cv::Scalar(200, 10, 10);  // blue
                break;
            default:
                throw std::runtime_error("Undefined type of prediction result");
        }
        int labelThickness = cellSize.width / 20;
        cv::Size labelTextSize = cv::getTextSize(label, fontType, 1, 2, &baseline);
        double labelFontScale = static_cast<double>(cellSize.width - 2 * labelThickness) / labelTextSize.width;
        cv::resize(mat, prevImg, cellSize);
        putHighlightedText(prevImg,
                           label,
                           cv::Point(labelThickness, cellSize.height - labelThickness - labelTextSize.height),
                           fontType,
                           labelFontScale,
                           textColor,
                           2);
        cv::Mat cell = outImg(cv::Rect(points[currSourceId], cellSize));
        prevImg.copyTo(cell);
        cv::rectangle(cell, {0, 0}, {cell.cols, cell.rows}, {255, 50, 50}, labelThickness);  // draw a border

        if (currSourceId == points.size() - 1) {
            currSourceId = 0;
        } else {
            currSourceId++;
        }
    }

private:
    cv::Mat prevImg;
    cv::Size cellSize;
    size_t currSourceId;
    std::vector<cv::Point> points;
    static const int fontType = cv::FONT_HERSHEY_PLAIN;
    static constexpr double fontScale = 1.5;
    static const int thickness = 2;
    static const int textPadding = 10;
    static constexpr const char testMessage[] = "Testing, please wait...";
    int baseline;
    cv::Size textSize;
    cv::Size accuracyMessageSize;
    cv::Size testMessageSize;
};

constexpr const char ClassificationGridMat::testMessage[];
