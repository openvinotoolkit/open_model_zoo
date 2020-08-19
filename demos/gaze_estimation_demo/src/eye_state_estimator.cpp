// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "eye_state_estimator.hpp"

namespace gaze_estimation {

EyeStateEstimator::EyeStateEstimator(InferenceEngine::Core& ie,
                                     const std::string& modelPath,
                                     const std::string& deviceName):
                                     ieWrapper(ie, modelPath, deviceName) {
    inputBlobName = ieWrapper.expectSingleInput();
    ieWrapper.expectImageInput(inputBlobName);
    outputBlobName = ieWrapper.expectSingleOutput();
}

cv::Rect EyeStateEstimator::createEyeBoundingBox(const cv::Point2i& p1,
                                                 const cv::Point2i& p2,
                                                 float scale) const {
    cv::Rect result;
    float size = static_cast<float>(cv::norm(p1 - p2));

    result.width = static_cast<int>(scale * size);
    result.height = result.width;

    auto midpoint = (p1 + p2) / 2;

    result.x = midpoint.x - (result.width / 2);
    result.y = midpoint.y - (result.height / 2);

    return result;
}

void EyeStateEstimator::rotateImageAroundCenter(const cv::Mat& srcImage,
                                                cv::Mat& dstImage,
                                                float angle) const {
    auto w = srcImage.cols;
    auto h = srcImage.rows;

    cv::Size size(w, h);

    cv::Point2f center(static_cast<float>(w / 2), static_cast<float>(h / 2));

    auto rotMatrix = cv::getRotationMatrix2D(center, static_cast<double>(angle), 1);
    cv::warpAffine(srcImage, dstImage, rotMatrix, size, 1, cv::BORDER_REPLICATE);
}

void EyeStateEstimator::estimate(const cv::Mat& image, FaceInferenceResults& outputResults) {
    auto roll = outputResults.headPoseAngles.z;

    outputResults.leftEyeMidpoint = (outputResults.faceLandmarks[0] + outputResults.faceLandmarks[1]) / 2;
    auto leftEyeBoundingBox = createEyeBoundingBox(outputResults.faceLandmarks[0], outputResults.faceLandmarks[1]);
    outputResults.leftEyeBoundingBox = leftEyeBoundingBox;
    if (leftEyeBoundingBox.area()) {
        auto leftEyeImage(cv::Mat(image, leftEyeBoundingBox));
        cv::Mat leftEyeImageRotated;
        rotateImageAroundCenter(leftEyeImage, leftEyeImageRotated, roll);
        leftEyeImage = leftEyeImageRotated;
        std::vector<float> outputValue;
        ieWrapper.setInputBlob(inputBlobName, leftEyeImage);
        ieWrapper.infer();
        ieWrapper.getOutputBlob(outputBlobName, outputValue);
        outputResults.leftEyeState = outputValue[0] < outputValue[1];
    } else {
        // Landmarks collapsed and the eye takes no area on image, pretend it's closed
        outputResults.leftEyeState = false;
    }

    outputResults.rightEyeMidpoint = (outputResults.faceLandmarks[2] + outputResults.faceLandmarks[3]) / 2;
    auto rightEyeBoundingBox = createEyeBoundingBox(outputResults.faceLandmarks[2], outputResults.faceLandmarks[3]);
    outputResults.rightEyeBoundingBox = rightEyeBoundingBox;
    if (rightEyeBoundingBox.area()) {
        auto rightEyeImage(cv::Mat(image, rightEyeBoundingBox));
        cv::Mat rightEyeImageRotated;
        rotateImageAroundCenter(rightEyeImage, rightEyeImageRotated, roll);
        rightEyeImage = rightEyeImageRotated;
        std::vector<float> outputValue;
        ieWrapper.setInputBlob(inputBlobName, rightEyeImage);
        ieWrapper.infer();
        ieWrapper.getOutputBlob(outputBlobName, outputValue);
        outputResults.rightEyeState = outputValue[0] < outputValue[1];
    } else {
        // Landmarks collapsed and the eye takes no area on image, pretend it's closed
        outputResults.rightEyeState = false;
    }
}

void EyeStateEstimator::printPerformanceCounts() const {
    ieWrapper.printPerlayerPerformance();
}

EyeStateEstimator::~EyeStateEstimator() {
}
}  // namespace gaze_estimation
