// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdio>
#include <string>
#include <vector>
#include <map>

#include "openvino/openvino.hpp"

#include "face_detector.hpp"

namespace gaze_estimation {

FaceDetector::FaceDetector(
    ov::Core& core, const std::string& modelPath, const std::string& deviceName,
    double detectionConfidenceThreshold, bool enableReshape) :
        ieWrapper(core, modelPath, modelType, deviceName),
        detectionThreshold(detectionConfidenceThreshold),
        enableReshape(enableReshape)
{
    const auto& inputInfo = ieWrapper.getInputTensorDimsInfo();

    inputTensorName = ieWrapper.expectSingleInput();
    ieWrapper.expectImageInput(inputTensorName);
    inputTensorDims = inputInfo.at(inputTensorName);

    const auto& outputInfo = ieWrapper.getOutputTensorDimsInfo();

    outputTensorName = ieWrapper.expectSingleOutput();
    const auto& outputTensorDims = outputInfo.at(outputTensorName);

    if (outputTensorDims.size() != 4 || outputTensorDims[0] != 1 || outputTensorDims[1] != 1 || outputTensorDims[3] != 7) {
        throw std::runtime_error(modelPath + ": expected \"" + outputTensorName + "\" to have shape 1x1xNx7");
    }

    numTotalDetections = outputTensorDims[2];
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
        double networkAspectRatio = std::round(100. * inputTensorDims[3] / inputTensorDims[2]) / 100.;
        double aspectRatioThreshold = 0.01;

         if (std::fabs(imageAspectRatio - networkAspectRatio) > aspectRatioThreshold) {
             slog::debug << "Face Detection network is reshaped" << slog::endl;
            // Fix height and change width to make networkAspectRatio equal to imageAspectRatio
            inputTensorDims[3] = static_cast<unsigned long>(inputTensorDims[2] * imageAspectRatio);

            ieWrapper.reshape({{inputTensorName, inputTensorDims}});
        }
    }

    ieWrapper.setInputTensor(inputTensorName, image);
    ieWrapper.infer();

    std::vector<float> rawDetectionResults;
    ieWrapper.getOutputTensor(outputTensorName, rawDetectionResults);
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

        // Ignore faces whose bounding boxes do not fit entirely into the frame
        if (rectIntersection.area() != faceRect.area())
            continue;

        tmp.faceDetectionConfidence = confidence;
        tmp.faceBoundingBox = faceRect;
        detectionResult.push_back(tmp);
    }

    return detectionResult;
}

FaceDetector::~FaceDetector() {
}

}  // namespace gaze_estimation
