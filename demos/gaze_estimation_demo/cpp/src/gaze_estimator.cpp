// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <cmath>

#include "gaze_estimator.hpp"

namespace gaze_estimation {

const char TENSOR_HEAD_POSE_ANGLES[] = "head_pose_angles";
const char TENSOR_LEFT_EYE_IMAGE[] = "left_eye_image";
const char TENSOR_RIGHT_EYE_IMAGE[] = "right_eye_image";

GazeEstimator::GazeEstimator(
    ov::Core& ie, const std::string& modelPath, const std::string& deviceName, bool doRollAlign) :
        ieWrapper(ie, modelPath, modelType, deviceName), rollAlign(doRollAlign)
{
    const auto& inputInfo = ieWrapper.getInputTensorDimsInfo();

    for (const auto& TensorName: {TENSOR_HEAD_POSE_ANGLES, TENSOR_LEFT_EYE_IMAGE, TENSOR_RIGHT_EYE_IMAGE}) {
        if (inputInfo.find(TensorName) == inputInfo.end())
            throw std::runtime_error(modelPath + ": expected to have input named \"" + TensorName + "\"");
    }

    auto expectAngles = [&modelPath](const std::string& TensorName, const ov::Shape dims) {
        bool is1Dim = !dims.empty()
            && std::all_of(dims.begin(), dims.end() - 1, [](unsigned long n) { return n == 1; });

        if (!is1Dim || dims.back() != 3) {
            throw std::runtime_error(modelPath + ": expected \"" + TensorName + "\" to have dimensions [1x...]3");
        }
    };

    expectAngles(TENSOR_HEAD_POSE_ANGLES, inputInfo.at(TENSOR_HEAD_POSE_ANGLES));

    for (const auto& TensorName: { TENSOR_LEFT_EYE_IMAGE, TENSOR_RIGHT_EYE_IMAGE}) {
        ieWrapper.expectImageInput(TensorName);
    }

    const auto& outputInfo = ieWrapper.getOutputTensorDimsInfo();

    outputTensorName = ieWrapper.expectSingleOutput();
    expectAngles(outputTensorName, outputInfo.at(outputTensorName));
}

void GazeEstimator::rotateImageAroundCenter(const cv::Mat& srcImage, cv::Mat& dstImage, float angle) const {
    auto w = srcImage.cols;
    auto h = srcImage.rows;

    cv::Size size(w, h);

    cv::Point2f center(static_cast<float>(w / 2), static_cast<float>(h / 2));

    auto rotMatrix = cv::getRotationMatrix2D(center, static_cast<double>(angle), 1);
    cv::warpAffine(srcImage, dstImage, rotMatrix, size, 1, cv::BORDER_REPLICATE);
}

void GazeEstimator::estimate(const cv::Mat& image, FaceInferenceResults& outputResults) {
    if (!outputResults.leftEyeState || !outputResults.rightEyeState)
        return;
    std::vector<float> headPoseAngles(3);
    auto roll = outputResults.headPoseAngles.z;
    headPoseAngles[0] = outputResults.headPoseAngles.x;
    headPoseAngles[1] = outputResults.headPoseAngles.y;
    headPoseAngles[2] = roll;

    cv::Mat leftEyeImage(image, outputResults.leftEyeBoundingBox);
    cv::Mat rightEyeImage(image, outputResults.rightEyeBoundingBox);

    if (rollAlign) {
        headPoseAngles[2] = 0;
        cv::Mat leftEyeImageRotated, rightEyeImageRotated;
        rotateImageAroundCenter(leftEyeImage, leftEyeImageRotated, roll);
        rotateImageAroundCenter(rightEyeImage, rightEyeImageRotated, roll);
        leftEyeImage = leftEyeImageRotated;
        rightEyeImage = rightEyeImageRotated;
    }

    ieWrapper.setInputTensor(TENSOR_HEAD_POSE_ANGLES, headPoseAngles);
    ieWrapper.setInputTensor(TENSOR_LEFT_EYE_IMAGE, leftEyeImage);
    ieWrapper.setInputTensor(TENSOR_RIGHT_EYE_IMAGE, rightEyeImage);

    ieWrapper.infer();

    std::vector<float> rawResults;

    ieWrapper.getOutputTensor(outputTensorName, rawResults);

    cv::Point3f gazeVector;
    gazeVector.x = rawResults[0];
    gazeVector.y = rawResults[1];
    gazeVector.z = rawResults[2];

    gazeVector = gazeVector / cv::norm(gazeVector);

    if (rollAlign) {
        // rotate gaze vector to compensate for the alignment
        float cs = static_cast<float>(std::cos(static_cast<double>(roll) * CV_PI / 180.0));
        float sn = static_cast<float>(std::sin(static_cast<double>(roll) * CV_PI / 180.0));

        auto tmpX = gazeVector.x * cs + gazeVector.y * sn;
        auto tmpY = -gazeVector.x * sn + gazeVector.y * cs;

        gazeVector.x = tmpX;
        gazeVector.y = tmpY;
    }

    outputResults.gazeVector = gazeVector;
}

GazeEstimator::~GazeEstimator() {
}
}  // namespace gaze_estimation
