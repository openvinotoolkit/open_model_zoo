// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

class GridMat {
public:
    cv::Mat outimg;

    explicit GridMat(const std::vector<cv::Size>& sizes, const cv::Size maxDisp = cv::Size{1920, 1080}) {
        size_t maxWidth = 0;
        size_t maxHeight = 0;
        for (size_t i = 0; i < sizes.size(); i++) {
            maxWidth = std::max(maxWidth, static_cast<size_t>(sizes[i].width));
            maxHeight = std::max(maxHeight, static_cast<size_t>(sizes[i].height));
        }
        if (0 == maxWidth || 0 == maxHeight) {
            throw std::invalid_argument("Input resolution must not be zero.");
        }

        size_t nGridCols = static_cast<size_t>(ceil(sqrt(static_cast<float>(sizes.size()))));
        size_t nGridRows = (sizes.size() - 1) / nGridCols + 1;
        size_t gridMaxWidth = static_cast<size_t>(maxDisp.width/nGridCols);
        size_t gridMaxHeight = static_cast<size_t>(maxDisp.height/nGridRows);

        float scaleWidth = static_cast<float>(gridMaxWidth) / maxWidth;
        float scaleHeight = static_cast<float>(gridMaxHeight) / maxHeight;
        float scaleFactor = std::min(1.f, std::min(scaleWidth, scaleHeight));

        cellSize.width = static_cast<int>(maxWidth * scaleFactor);
        cellSize.height = static_cast<int>(maxHeight * scaleFactor);

        for (size_t i = 0; i < sizes.size(); i++) {
            cv::Point p;
            p.x = cellSize.width * (i % nGridCols);
            p.y = cellSize.height * (i / nGridCols);
            points.push_back(p);
        }

        outimg.create(cellSize.height * nGridRows, cellSize.width * nGridCols, CV_8UC3);
        outimg.setTo(0);
        clear();
    }

    cv::Size getCellSize() {
        return cellSize;
    }

    void fill(std::vector<cv::Mat>& frames) {
        if (frames.size() > points.size()) {
            throw std::logic_error("Cannot display " + std::to_string(frames.size()) + " channels in a grid with " + std::to_string(points.size()) + " cells");
        }

        for (size_t i = 0; i < frames.size(); i++) {
            cv::Mat cell = outimg(cv::Rect(points[i].x, points[i].y, cellSize.width, cellSize.height));

            if ((cellSize.width == frames[i].cols) && (cellSize.height == frames[i].rows)) {
                frames[i].copyTo(cell);
            } else if ((cellSize.width > frames[i].cols) && (cellSize.height > frames[i].rows)) {
                frames[i].copyTo(cell(cv::Rect(0, 0, frames[i].cols, frames[i].rows)));
            } else {
                cv::resize(frames[i], cell, cellSize);
            }
        }
        unupdatedSourceIDs.clear();
    }

    void update(const cv::Mat& frame, const size_t sourceID) {
        const cv::Mat& cell = outimg(cv::Rect(points[sourceID], cellSize));

        if ((cellSize.width == frame.cols) && (cellSize.height == frame.rows)) {
            frame.copyTo(cell);
        } else if ((cellSize.width > frame.cols) && (cellSize.height > frame.rows)) {
            frame.copyTo(cell(cv::Rect(0, 0, frame.cols, frame.rows)));
        } else {
            cv::resize(frame, cell, cellSize);
        }
        unupdatedSourceIDs.erase(unupdatedSourceIDs.find(sourceID));
    }

    bool isFilled() const noexcept {
        return unupdatedSourceIDs.empty();
    }
    void clear() {
        size_t counter = 0;
        std::generate_n(std::inserter(unupdatedSourceIDs, unupdatedSourceIDs.end()), points.size(), [&counter]{return counter++;});
    }
    std::set<size_t> getUnupdatedSourceIDs() const noexcept {
        return unupdatedSourceIDs;
    }
    cv::Mat getMat() const noexcept {
        return outimg;
    }

private:
    cv::Size cellSize;
    std::set<size_t> unupdatedSourceIDs;
    std::vector<cv::Point> points;
};

void fillROIColor(cv::Mat& displayImage, cv::Rect roi, cv::Scalar color, double opacity) {
    if (opacity > 0) {
        roi = roi & cv::Rect(0, 0, displayImage.cols, displayImage.rows);
        cv::Mat textROI = displayImage(roi);
        cv::addWeighted(color, opacity, textROI, 1.0 - opacity , 0.0, textROI);
    }
}

void putTextOnImage(cv::Mat& displayImage, std::string str, cv::Point p,
                    cv::HersheyFonts font, double fontScale, cv::Scalar color,
                    int thickness = 1, cv::Scalar bgcolor = cv::Scalar(),
                    double opacity = 0) {
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(str, font, 0.5, 1, &baseline);
    fillROIColor(displayImage, cv::Rect(cv::Point(p.x, p.y + baseline),
                                        cv::Point(p.x + textSize.width, p.y - textSize.height)),
                 bgcolor, opacity);
    cv::putText(displayImage, str, p, font, fontScale, color, thickness);
}
