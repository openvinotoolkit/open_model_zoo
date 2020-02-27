// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <cmath>

#include "gaze_estimator.hpp"

namespace gaze_estimation {

const char BLOB_HEAD_POSE_ANGLES[] = "head_pose_angles";
const char BLOB_LEFT_EYE_IMAGE[] = "left_eye_image";
const char BLOB_RIGHT_EYE_IMAGE[] = "right_eye_image";

GazeEstimator::GazeEstimator(InferenceEngine::Core& ie,
                             const std::string& modelPath,
                             const std::string& deviceName,
                             bool doRollAlign):
               ieWrapper(ie, modelPath, deviceName), rollAlign(doRollAlign) {
    const auto& inputInfo = ieWrapper.getInputBlobDimsInfo();

    for (const auto& blobName: {BLOB_HEAD_POSE_ANGLES, BLOB_LEFT_EYE_IMAGE, BLOB_RIGHT_EYE_IMAGE}) {
        if (inputInfo.find(blobName) == inputInfo.end())
            throw std::runtime_error(modelPath + ": expected to have input named \"" + blobName + "\"");
    }

    auto expectAngles = [&modelPath](const std::string& blobName, const std::vector<unsigned long>& dims) {
        bool is1Dim = !dims.empty()
            && std::all_of(dims.begin(), dims.end() - 1, [](unsigned long n) { return n == 1; });

        if (!is1Dim || dims.back() != 3) {
            throw std::runtime_error(modelPath + ": expected \"" + blobName + "\" to have dimensions [1x...]3");
        }
    };

    expectAngles(BLOB_HEAD_POSE_ANGLES, inputInfo.at(BLOB_HEAD_POSE_ANGLES));

    for (const auto& blobName: {BLOB_LEFT_EYE_IMAGE, BLOB_RIGHT_EYE_IMAGE}) {
        ieWrapper.expectImageInput(blobName);
    }

    const auto& outputInfo = ieWrapper.getOutputBlobDimsInfo();

    outputBlobName = ieWrapper.expectSingleOutput();
    expectAngles(outputBlobName, outputInfo.at(outputBlobName));
}

cv::Rect GazeEstimator::createEyeBoundingBox(const cv::Point2i& p1,
                                             const cv::Point2i& p2,
                                             float scale) const {
    cv::Rect result;
    float size = static_cast<float>(cv::norm(p1-p2));

    result.width = static_cast<int>(scale * size);
    result.height = result.width;

    auto midpoint = (p1 + p2) / 2;

    result.x = midpoint.x - (result.width / 2);
    result.y = midpoint.y - (result.height / 2);

    return result;
}

void GazeEstimator::rotateImageAroundCenter(const cv::Mat& srcImage,
                                              cv::Mat& dstImage,
                                              float angle) const {
    auto w = srcImage.cols;
    auto h = srcImage.rows;

    cv::Size size(w, h);

    cv::Point2f center(static_cast<float>(w / 2), static_cast<float>(h / 2));

    auto rotMatrix = cv::getRotationMatrix2D(center, static_cast<double>(angle), 1);
    cv::warpAffine(srcImage, dstImage, rotMatrix, size, 1, cv::BORDER_REPLICATE);
}


void GazeEstimator::estimate(const cv::Mat& image,
                             FaceInferenceResults& outputResults) {
    std::vector<float> headPoseAngles(3);
    auto roll = outputResults.headPoseAngles.z;
    headPoseAngles[0] = outputResults.headPoseAngles.x;
    headPoseAngles[1] = outputResults.headPoseAngles.y;
    headPoseAngles[2] = roll;

    auto leftEyeBoundingBox = createEyeBoundingBox(outputResults.faceLandmarks[0],
                                                     outputResults.faceLandmarks[1]);
    auto rightEyeBoundingBox = createEyeBoundingBox(outputResults.faceLandmarks[2],
                                                      outputResults.faceLandmarks[3]);

    auto leftEyeMidpoint = (outputResults.faceLandmarks[0] + outputResults.faceLandmarks[1]) / 2;
    auto rightEyeMidpoint = (outputResults.faceLandmarks[2] + outputResults.faceLandmarks[3]) / 2;

    outputResults.leftEyeBoundingBox = leftEyeBoundingBox;
    outputResults.rightEyeBoundingBox = rightEyeBoundingBox;

    outputResults.leftEyeMidpoint = leftEyeMidpoint;
    outputResults.rightEyeMidpoint = rightEyeMidpoint;

    cv::Mat leftEyeImage(image, leftEyeBoundingBox);
    cv::Mat rightEyeImage(image, rightEyeBoundingBox);

    if (rollAlign) {
        headPoseAngles[2] = 0;
        cv::Mat leftEyeImageRotated, rightEyeImageRotated;
        rotateImageAroundCenter(leftEyeImage, leftEyeImageRotated, roll);
        rotateImageAroundCenter(rightEyeImage, rightEyeImageRotated, roll);
        leftEyeImage = leftEyeImageRotated;
        rightEyeImage = rightEyeImageRotated;
    }

    ieWrapper.setInputBlob(BLOB_HEAD_POSE_ANGLES, headPoseAngles);
    ieWrapper.setInputBlob(BLOB_LEFT_EYE_IMAGE, leftEyeImage);
    ieWrapper.setInputBlob(BLOB_RIGHT_EYE_IMAGE, rightEyeImage);

    ieWrapper.infer();

    std::vector<float> rawResults;

    ieWrapper.getOutputBlob(outputBlobName, rawResults);

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

void GazeEstimator::printPerformanceCounts() const {
    ieWrapper.printPerlayerPerformance();
}

GazeEstimator::~GazeEstimator() {
}
}  // namespace gaze_estimation
