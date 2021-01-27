/*
// Copyright (C) 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <algorithm>
#include <string>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>

#include "models/associative_embedding_decoder.h"
#include "models/hpe_model_associative_embedding.h"

#include <utils/common.hpp>
#include <utils/ocv_common.hpp>
#include <ngraph/ngraph.hpp>

using namespace InferenceEngine;

const cv::Vec3f HpeAssociativeEmbedding::meanPixel = cv::Vec3f::all(128);
const float HpeAssociativeEmbedding::detectionThreshold = 0.1f;
const float HpeAssociativeEmbedding::tagThreshold = 1.0f;
const float HpeAssociativeEmbedding::delta = 0.0f;
const bool HpeAssociativeEmbedding::useDetectionVal = true;
const bool HpeAssociativeEmbedding::doAdjust = true;
const bool HpeAssociativeEmbedding::doRefine = true;
const bool HpeAssociativeEmbedding::ignoreTooMuch = true;

HpeAssociativeEmbedding::HpeAssociativeEmbedding(const std::string& modelFileName, double aspectRatio,
    int targetSize, float confidenceThreshold) :
    ModelBase(modelFileName),
    aspectRatio(aspectRatio),
    targetSize(targetSize),
    confidenceThreshold(confidenceThreshold) {
}

void HpeAssociativeEmbedding::prepareInputsOutputs(CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input blobs ------------------------------------------------------
    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    if (inputShapes.size() != 1)
        throw std::runtime_error("Demo supports topologies only with 1 input");
    inputsNames.push_back(inputShapes.begin()->first);
    SizeVector& inSizeVector = inputShapes.begin()->second;
    if (inSizeVector.size() != 4 || inSizeVector[0] != 1 || inSizeVector[1] != 3)
        throw std::runtime_error("3-channel 4-dimensional model's input is expected");
    InputInfo& inputInfo = *cnnNetwork.getInputsInfo().begin()->second;
    inputInfo.setPrecision(Precision::U8);
    inputInfo.getInputData()->setLayout(Layout::NHWC);

    // --------------------------- Prepare output blobs -----------------------------------------------------
    const OutputsDataMap& outputInfo = cnnNetwork.getOutputsInfo();
    if (outputInfo.size() != 3)
        throw std::runtime_error("Demo supports topologies only with 3 outputs");
    for (const auto& outputLayer: outputInfo) {
        outputLayer.second->setPrecision(Precision::FP32);
        outputsNames.push_back(outputLayer.first);
        const SizeVector& outputLayerDims = outputLayer.second->getTensorDesc().getDims();
        if (outputLayerDims.size() != 4 && outputLayerDims.size() != 5)
                throw std::runtime_error("output layers are expected to be 4-dimensional or 5-dimensional");
        if (outputLayerDims[0] != 1 || outputLayerDims[1] != 17)
                throw std::runtime_error("output layers are expected to have 1 batch size and 17 channels");
    }
    embeddingsBlobName = findLayerByName("embeddings", outputsNames);
    heatmapsBlobName = findLayerByName("heatmaps", outputsNames);
    nmsHeatmapsBlobName = findLayerByName("nms_heatmaps", outputsNames);
}

void HpeAssociativeEmbedding::reshape(CNNNetwork& cnnNetwork) {
    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    SizeVector& inputDims = inputShapes.begin()->second;
    if (!targetSize) {
        targetSize =  static_cast<int>(std::min(inputDims[2], inputDims[3]));
    }
    int inputHeight = static_cast<int>(std::round(targetSize * aspectRatio));
    int inputWidth = targetSize;
    if (aspectRatio >= 1.0) {
        std::swap(inputHeight, inputWidth);
    }
    int height = static_cast<int>((inputHeight + stride - 1) / stride) * stride;
    int width = static_cast<int>((inputWidth + stride - 1) / stride) * stride;
    inputDims[0] = 1;
    inputDims[2] = height;
    inputDims[3] = width;
    inputLayerSize = cv::Size(inputDims[3], inputDims[2]);
    cnnNetwork.reshape(inputShapes);
}

std::shared_ptr<InternalModelData> HpeAssociativeEmbedding::preprocess(const InputData& inputData, InferRequest::Ptr& request) {
    auto& image = inputData.asRef<ImageInputData>().inputImage;
    cv::Mat resizedImage;
    float scale = std::min(inputLayerSize.height / static_cast<float>(image.rows),
                           inputLayerSize.width / static_cast<float>(image.cols));
    cv::resize(image, resizedImage, cv::Size(), scale, scale, cv::INTER_CUBIC);
    int h = resizedImage.rows;
    int w = resizedImage.cols;
    cv::Mat paddedImage;
    int bottom = inputLayerSize.height - h;
    int right = inputLayerSize.width - w;
    cv::copyMakeBorder(resizedImage, paddedImage, 0, bottom, 0, right,
                       cv::BORDER_CONSTANT, meanPixel);
    request->SetBlob(inputsNames[0], wrapMat2Blob(paddedImage));
    return std::shared_ptr<InternalModelData>(new InternalScaleData(image.cols / static_cast<float>(w),
                                                                    image.rows / static_cast<float>(h)));
}

std::unique_ptr<ResultBase> HpeAssociativeEmbedding::postprocess(InferenceResult& infResult) {
    HumanPoseResult* result = new HumanPoseResult;
    *static_cast<ResultBase*>(result) = static_cast<ResultBase&>(infResult);

    auto aembds = infResult.outputsData[embeddingsBlobName];
    auto heats = infResult.outputsData[heatmapsBlobName];
    auto nmsHeats = infResult.outputsData[nmsHeatmapsBlobName];

    const SizeVector& aembdsDims = aembds->getTensorDesc().getDims();
    const SizeVector& heatMapsDims = heats->getTensorDesc().getDims();
    const SizeVector& nmsHeatMapsDims = nmsHeats->getTensorDesc().getDims();

    float* aembdsMapped = aembds->rmap().as<float*>();
    float* heatMapsMapped = heats->rmap().as<float*>();
    float* nmsHeatMapsMapped = nmsHeats->rmap().as<float*>();

    std::vector<cv::Mat>& aembdsMaps = split(aembdsMapped, aembdsDims);
    std::vector<cv::Mat>& heatMaps = split(heatMapsMapped, heatMapsDims);
    std::vector<cv::Mat>& nmsHeatMaps = split(nmsHeatMapsMapped, nmsHeatMapsDims);

    std::vector<HumanPose> poses = extractPoses(heatMaps, aembdsMaps, nmsHeatMaps);

    // Rescale poses to the original image
    const auto& scale = infResult.internalModelData->asRef<InternalScaleData>();
    float outputScale = inputLayerSize.width / static_cast<float>(heatMapsDims[3]);
    float scaleX = scale.x * outputScale;
    float scaleY = scale.y * outputScale;

    for (auto& pose : poses) {
        for (auto& keypoint : pose.keypoints) {
            if (keypoint != cv::Point2f(-1, -1)) {
                keypoint.x *= scaleX;
                keypoint.y *= scaleY;
                std::swap(keypoint.x, keypoint.y);
            }
        }
        result->poses.push_back(pose);
    }

    return std::unique_ptr<ResultBase>(result);
}

std::string HpeAssociativeEmbedding::findLayerByName(const std::string layerName,
                                                     const std::vector<std::string>& outputsNames) {
    std::vector<std::string> suitableLayers;
    for (auto& outputName: outputsNames) {
        if (outputName.rfind(layerName, 0) == 0) {
           suitableLayers.push_back(outputName);
        }
    }
    if (suitableLayers.empty())
        throw std::runtime_error("Suitable layer for " + layerName + " output is not found");
    else if (suitableLayers.size() > 1)
        throw std::runtime_error("More than 1 layer matched to " + layerName + " output");
    return suitableLayers[0];
}

std::vector<cv::Mat> HpeAssociativeEmbedding::split(float* data, const SizeVector& shape) {
    std::vector<cv::Mat> flattenData(shape[1]);
    for (size_t i = 0; i < flattenData.size(); i++) {
        flattenData[i] = cv::Mat(shape[2], shape[3], CV_32FC1, data + i * shape[2] * shape[3]);
    }
    return flattenData;
}

std::vector<HumanPose> HpeAssociativeEmbedding::extractPoses(
    const std::vector<cv::Mat>& heatMaps,
    const std::vector<cv::Mat>& aembdsMaps,
    const std::vector<cv::Mat>& nmsHeatMaps) const {

    std::vector<std::vector<Peak>> allPeaks(numJoints);
    for (int i = 0; i < numJoints; i++) {
        findPeaks(nmsHeatMaps, aembdsMaps, allPeaks, i, maxNumPeople, detectionThreshold);
    }
    std::vector<Pose> allPoses = matchByTag(allPeaks, maxNumPeople, numJoints,
                                            tagThreshold, useDetectionVal, ignoreTooMuch);
    std::vector<HumanPose> poses;
    for (size_t i = 0; i < allPoses.size(); i++) {
        Pose pose = allPoses[i];
        // Filtering poses with low mean scores
        if (pose.getMeanScore() <= confidenceThreshold) {
            continue;
        }
        if (doAdjust || doRefine) {
            adjustAndRefine(allPoses, heatMaps, aembdsMaps, i, delta, doAdjust, doRefine);
        }
        std::vector<cv::Point2f> keypoints;
        for (size_t j = 0; j < numJoints; j++) {
            Peak& peak = pose.getPeak(j);
            keypoints.push_back(peak.keypoint);
        }
        poses.push_back({keypoints, pose.getMeanScore()});
    }
    return poses;
}
