// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdio>
#include <string>

#include <vector>
#include <map>

#include "face_detector.hpp"

namespace gaze_estimation {
FaceDetector::FaceDetector(InferenceEngine::Core& ie,
                           const std::string& modelPath,
                           const std::string& deviceName,
                           double detectionConfidenceThreshold,
                           bool enableReshape):
             ieWrapper(ie, modelPath, deviceName),
             detectionThreshold(detectionConfidenceThreshold),
             enableReshape(enableReshape) {
    const auto& inputInfo = ieWrapper.getInputBlobDimsInfo();

    inputBlobName = ieWrapper.expectSingleInput();
    ieWrapper.expectImageInput(inputBlobName);
    inputBlobDims = inputInfo.at(inputBlobName);

    const auto& outputInfo = ieWrapper.getOutputBlobDimsInfo();

    outputBlobName = ieWrapper.expectSingleOutput();
    const auto& outputBlobDims = outputInfo.at(outputBlobName);

    if (outputBlobDims.size() != 4 || outputBlobDims[0] != 1 || outputBlobDims[1] != 1 || outputBlobDims[3] != 7) {
        throw std::runtime_error(modelPath + ": expected \"" + outputBlobName + "\" to have shape 1x1xNx7");
    }

    numTotalDetections = outputBlobDims[2];
}

void FaceDetector::adjustBoundingBox(cv::Rect& boundingBox) const {
    auto w = boundingBox.width;
    auto h = boundingBox.height;

    boundingBox.x -= static_cast<int>(0.067 * w);
    boundingBox.y -= static_cast<int>(0.028 * h);

    boundingBox.width += static_cast<int>(0.15 * w);
    boundingBox.height += static_cast<int>(0.13 * h);

    if (boundingBox.width < boundingBox.height) {
        auto dx = (boundingBox.height - boundingBox.width);
        boundingBox.x -= dx / 2;
        boundingBox.width += dx;
    } else {
        auto dy = (boundingBox.width - boundingBox.height);
        boundingBox.y -= dy / 2;
        boundingBox.height += dy;
    }
}

std::vector<FaceInferenceResults> FaceDetector::detect(const cv::Mat& image) {
    std::vector<FaceInferenceResults> detectionResult;

    if (enableReshape) {
        double imageAspectRatio = std::round(100. * image.cols / image.rows) / 100.;
        double networkAspectRatio = std::round(100. * inputBlobDims[3] / inputBlobDims[2]) / 100.;
        double aspectRatioThreshold = 0.01;

         if (std::fabs(imageAspectRatio - networkAspectRatio) > aspectRatioThreshold) {
            std::cout << "Face Detection network is reshaped" << std::endl;

            // Fix height and change width to make networkAspectRatio equal to imageAspectRatio
            inputBlobDims[3] = static_cast<unsigned long>(inputBlobDims[2] * imageAspectRatio);

            ieWrapper.reshape({{inputBlobName, inputBlobDims}});
        }
    }

    ieWrapper.setInputBlob(inputBlobName, image);
    ieWrapper.infer();

    std::vector<float> rawDetectionResults;
    ieWrapper.getOutputBlob(outputBlobName, rawDetectionResults);

    FaceInferenceResults tmp;

    cv::Size imageSize(image.size());
    cv::Rect imageRect(0, 0, image.cols, image.rows);

    for (unsigned long detectionID = 0; detectionID < numTotalDetections; ++detectionID) {
        float confidence = rawDetectionResults[detectionID * 7 + 2];
        if (static_cast<double>(confidence) < detectionThreshold) {
            break;
        }

        auto x = rawDetectionResults[detectionID * 7 + 3] * imageSize.width;
        auto width = rawDetectionResults[detectionID * 7 + 5] * imageSize.width - x;
        auto y = rawDetectionResults[detectionID * 7 + 4] * imageSize.height;
        auto height = rawDetectionResults[detectionID * 7 + 6] * imageSize.height - y;

        cv::Rect faceRect(static_cast<int>(x), static_cast<int>(y),
                          static_cast<int>(width), static_cast<int>(height));
        adjustBoundingBox(faceRect);

        auto rectIntersection = faceRect & imageRect;

        // Ignore faces whose bouding boxes do not fit entirely into the frame
        if (rectIntersection.area() != faceRect.area())
            continue;

        tmp.faceDetectionConfidence = confidence;
        tmp.faceBoundingBox = faceRect;
        detectionResult.push_back(tmp);
    }

    return detectionResult;
}

void FaceDetector::printPerformanceCounts() const {
    ieWrapper.printPerlayerPerformance();
}

FaceDetector::~FaceDetector() {
}

}  // namespace gaze_estimation
