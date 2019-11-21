// Copyright (C) 2018 Intel Corporation
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
    explicit GridMat(const cv::Size maxDisp = cv::Size{1920, 1080},
                     const cv::Size cellSize = cv::Size{240, 135},
                     size_t rh = 60):
    cellSize{cellSize}, currSourceID{0}, rectangleHeight{rh} {
        size_t cNum = ceil(maxDisp.width / cellSize.width);
        size_t rNum = ceil(maxDisp.height / cellSize.height);
        positionNum = cNum * rNum;

        for (size_t i = 0; i < cNum * rNum; i++) {
            cv::Point p;
            p.x = cellSize.width * (i % cNum);
            p.y = rectangleHeight + (cellSize.height * (i / cNum));
            points.push_back(p);
        }

        outImg.create((cellSize.height * rNum) + rectangleHeight, cellSize.width * cNum, CV_8UC3);
        outImg.setTo(0);
    }

    cv::Size getCellSize() {
        return cellSize;
    }

    void listUpdate(std::queue<cv::Mat>& frames) {
        while(!frames.empty()) {
            updateList.push_back(frames.front());
            frames.pop();
        }
    }

    void textUpdate(double overSPF){
        //set rectangle
        size_t colunmNum = outImg.cols;
        cv::Point p1 = cv::Point(0,0);
        cv::Point p2 = cv::Point(colunmNum, rectangleHeight);
        
        rectangle(outImg,p1,p2,
            cv::Scalar(0,0,0), cv::FILLED);
        
        //set text        
        auto frameWidth = outImg.cols;
        double fontScale = frameWidth * 1. / 640;
        auto fontColor = cv::Scalar(0, 255, 0);
        int thickness = 2;

        cv::putText(outImg,
                    cv::format("Overall FPS: %0.01f", 1./overSPF),
                    cv::Point(10, static_cast<int>(30 * fontScale / 1.6)),
                    cv::FONT_HERSHEY_PLAIN, fontScale, fontColor, thickness);
    }

    cv::Mat getMat() {
        while(!updateList.empty()) {    
            cv::Mat cell = outImg(cv::Rect(points[currSourceID], cellSize));
            cv::Mat frame = updateList.front();
            updateList.pop_front();

            if ((cellSize.width == frame.cols) && (cellSize.height == frame.rows)) {
                frame.copyTo(cell);
            } else if ((cellSize.width > frame.cols) && (cellSize.height > frame.rows)) {
                frame.copyTo(cell(cv::Rect(0, 0, frame.cols, frame.rows)));
            } else {
                cv::resize(frame, cell, cellSize);
            }
            
            if(currSourceID == points.size() - 1)
                currSourceID = 0;
            else
                currSourceID++;
        }

        return outImg;
    }

private:
    cv::Mat outImg;
    std::list<cv::Mat> updateList;
    cv::Size cellSize;
    size_t currSourceID;
    std::vector<cv::Point> points;
    size_t rectangleHeight;
    size_t positionNum;
};
