// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cstdio>
#include <iostream>

#define _USE_MATH_DEFINES
#include <cmath>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include "results_marker.hpp"
#include "face_inference_results.hpp"
#include "utils.hpp"

namespace gaze_estimation {
ResultsMarker::ResultsMarker(bool showFaceBoundingBox,
                             bool showHeadPoseAxes,
                             bool showLandmarks,
                             bool showGaze):
                             showFaceBoundingBox(showFaceBoundingBox),
                             showHeadPoseAxes(showHeadPoseAxes),
                             showLandmarks(showLandmarks),
                             showGaze(showGaze) {
}

void ResultsMarker::mark(cv::Mat& image,
                         const FaceInferenceResults& faceInferenceResults) const {
    auto faceBoundingBox = faceInferenceResults.faceBoundingBox;
    auto faceBoundingBoxWidth = faceBoundingBox.width;
    auto faceBoundingBoxHeight = faceBoundingBox.height;
    auto scale =  0.002 * faceBoundingBoxWidth;
    cv::Point tl = faceBoundingBox.tl();

    if (showFaceBoundingBox) {
        cv::rectangle(image, faceInferenceResults.faceBoundingBox, cv::Scalar::all(255), 1);
        cv::putText(image,
                    cv::format("Detector confidence: %0.2f",
                               static_cast<double>(faceInferenceResults.faceDetectionConfidence)),
                    cv::Point(static_cast<int>(tl.x),
                              static_cast<int>(tl.y - 5. * faceBoundingBoxWidth / 200.)),
                    cv::FONT_HERSHEY_COMPLEX, scale, cv::Scalar::all(255), 1);
    }

    if (showHeadPoseAxes) {
        auto yaw = static_cast<double>(faceInferenceResults.headPoseAngles.x);
        auto pitch = static_cast<double>(faceInferenceResults.headPoseAngles.y);
        auto roll = static_cast<double>(faceInferenceResults.headPoseAngles.z);

        auto sinY = std::sin(yaw * M_PI / 180.0);
        auto sinP = std::sin(pitch * M_PI / 180.0);
        auto sinR = std::sin(roll * M_PI / 180.0);

        auto cosY = std::cos(yaw * M_PI / 180.0);
        auto cosP = std::cos(pitch * M_PI / 180.0);
        auto cosR = std::cos(roll * M_PI / 180.0);

        auto axisLength = 0.4 * faceBoundingBoxWidth;
        auto xCenter = faceBoundingBox.x + faceBoundingBoxWidth / 2;
        auto yCenter = faceBoundingBox.y + faceBoundingBoxHeight / 2;

        // center to right
        cv::line(image, cv::Point(xCenter, yCenter),
                 cv::Point(static_cast<int>(xCenter + axisLength * (cosR * cosY + sinY * sinP * sinR)),
                           static_cast<int>(yCenter + axisLength * cosP * sinR)),
                 cv::Scalar(0, 0, 255), 2);
        // center to top
        cv::line(image, cv::Point(xCenter, yCenter),
                 cv::Point(static_cast<int>(xCenter + axisLength * (cosR * sinY * sinP + cosY * sinR)),
                           static_cast<int>(yCenter - axisLength * cosP * cosR)),
                 cv::Scalar(0, 255, 0), 2);
        // center to forward
        cv::line(image, cv::Point(xCenter, yCenter),
                 cv::Point(static_cast<int>(xCenter + axisLength * sinY * cosP),
                           static_cast<int>(yCenter + axisLength * sinP)),
                 cv::Scalar(255, 0, 255), 2);

        cv::putText(image,
                    cv::format("head pose: (y=%0.0f, p=%0.0f, r=%0.0f)", std::round(yaw), std::round(pitch), std::round(roll)),
                    cv::Point(static_cast<int>(faceBoundingBox.tl().x),
                              static_cast<int>(faceBoundingBox.br().y + 5. * faceBoundingBoxWidth / 100.)),
                    cv::FONT_HERSHEY_PLAIN, scale * 2, cv::Scalar(255, 255, 255), 1);
    }

    if (showLandmarks) {
        int lmRadius = static_cast<int>(0.01 * faceBoundingBoxWidth + 1);
        for (auto const& point : faceInferenceResults.faceLandmarks)
            cv::circle(image, point, lmRadius, cv::Scalar(0, 255, 255), -1);
    }

    if (showGaze) {
        auto gazeVector = faceInferenceResults.gazeVector;

        double arrowLength = 0.4 * faceBoundingBoxWidth;
        cv::Point2f gazeArrow;
        gazeArrow.x = gazeVector.x;
        gazeArrow.y = -gazeVector.y;

        gazeArrow = arrowLength * gazeArrow;

        // Draw eyes bounding boxes
        cv::rectangle(image, faceInferenceResults.leftEyeBoundingBox, cv::Scalar::all(255), 1);
        cv::rectangle(image, faceInferenceResults.rightEyeBoundingBox, cv::Scalar::all(255), 1);

        cv::arrowedLine(image,
            faceInferenceResults.leftEyeMidpoint,
            faceInferenceResults.leftEyeMidpoint + gazeArrow, cv::Scalar(255, 0, 0), 2);

        cv::arrowedLine(image,
            faceInferenceResults.rightEyeMidpoint,
            faceInferenceResults.rightEyeMidpoint + gazeArrow, cv::Scalar(255, 0, 0), 2);

        cv::Point2f gazeAngles;

        gazeVectorToGazeAngles(faceInferenceResults.gazeVector, gazeAngles);

        cv::putText(image,
                    cv::format("gaze angles: (h=%0.0f, v=%0.0f)",
                               static_cast<double>(std::round(gazeAngles.x)),
                               static_cast<double>(std::round(gazeAngles.y))),
                    cv::Point(static_cast<int>(faceBoundingBox.tl().x),
                              static_cast<int>(faceBoundingBox.br().y + 12. * faceBoundingBoxWidth / 100.)),
                    cv::FONT_HERSHEY_PLAIN, scale * 2, cv::Scalar::all(255), 1);
    }
}

void ResultsMarker::toggle(char key) {
    if (key == 'l') {
        showLandmarks = !showLandmarks;
    } else if (key == 'h') {
        showHeadPoseAxes = !showHeadPoseAxes;
    } else if (key == 'g') {
        showGaze = !showGaze;
    } else if (key == 'd') {
        showFaceBoundingBox = !showFaceBoundingBox;
    } else if (key == 'a') {
        showFaceBoundingBox = true;
        showHeadPoseAxes = true;
        showLandmarks = true;
        showGaze = true;
    } else if (key == 'n') {
        showFaceBoundingBox = false;
        showHeadPoseAxes = false;
        showLandmarks = false;
        showGaze = false;
    }
}
}  // namespace gaze_estimation
