// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

#include "face.hpp"

// -------------------------Generic routines for visualization of detection results-------------------------------------------------

// Drawing a bar of emotions
class EmotionBarVisualizer {
public:
    using Ptr = std::shared_ptr<EmotionBarVisualizer>;

    explicit EmotionBarVisualizer(std::vector<std::string> const& emotionNames, cv::Size size = cv::Size(300, 140), cv::Size padding = cv::Size(10, 10),
                              double opacity = 0.6, double textScale = 1, int textThickness = 1);

    void draw(cv::Mat& img, std::map<std::string, float> emotions, cv::Point org, cv::Scalar fgcolor, cv::Scalar bgcolor);
    cv::Size getSize();
private:
    std::vector<std::string> emotionNames;
    cv::Size size;
    cv::Size padding;
    cv::Size textSize;
    int textBaseline;
    int ystep;
    double opacity;
    double textScale;
    int textThickness;
    int internalPadding;
};

// Drawing a photo frame around detected face
class PhotoFrameVisualizer {
public:
    using Ptr = std::shared_ptr<PhotoFrameVisualizer>;

    explicit PhotoFrameVisualizer(int bbThickness = 1, int photoFrameThickness = 2, float photoFrameLength = 0.1);

    void draw(cv::Mat& img, cv::Rect& bb, cv::Scalar color);

private:
    int bbThickness;
    int photoFrameThickness;
    float photoFrameLength;
};

// Drawing the position of the head
class HeadPoseVisualizer {
public:
    using Ptr = std::shared_ptr<HeadPoseVisualizer>;

    explicit HeadPoseVisualizer(float scale = 50,
                            cv::Scalar xAxisColor = cv::Scalar(0, 0, 255),
                            cv::Scalar yAxisColor = cv::Scalar(0, 255, 0),
                            cv::Scalar zAxisColor = cv::Scalar(255, 0, 0),
                            int axisThickness = 2);

    void draw(cv::Mat& frame, cv::Point3f cpoint, HeadPoseDetection::Results headPose);

private:
    void buildCameraMatrix(cv::Mat& cameraMatrix, int cx, int cy, float focalLength);

private:
    cv::Scalar xAxisColor;
    cv::Scalar yAxisColor;
    cv::Scalar zAxisColor;
    int axisThickness;
    float scale;
};

// Drawing detected faces on the frame
class Visualizer {
public:
    using Ptr = std::shared_ptr<Visualizer>;

    enum AnchorType {
        TL = 0,
        TR,
        BL,
        BR
    };

    struct DrawParams {
        cv::Point cell;
        AnchorType barAnchor;
        AnchorType rectAnchor;
        size_t frameIdx;
    };

    explicit Visualizer(cv::Size const& imgSize, int leftPadding = 10, int rightPadding = 10, int topPadding = 75, int bottomPadding = 10);

    void enableEmotionBar(std::vector<std::string> const& emotionNames);
    void draw(cv::Mat img, std::list<Face::Ptr> faces);

private:
    void drawFace(cv::Mat& img, Face::Ptr f, bool drawEmotionBar);
    cv::Point findCellForEmotionBar();

    std::map<size_t, DrawParams> drawParams;
    EmotionBarVisualizer::Ptr emotionVisualizer;
    PhotoFrameVisualizer::Ptr photoFrameVisualizer;
    HeadPoseVisualizer::Ptr headPoseVisualizer;

    cv::Mat drawMap;
    int nxcells;
    int nycells;
    int xstep;
    int ystep;

    cv::Size imgSize;
    int leftPadding;
    int rightPadding;
    int topPadding;
    int bottomPadding;
    cv::Size emotionBarSize;
    size_t frameCounter;
};
