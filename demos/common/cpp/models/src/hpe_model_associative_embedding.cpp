///*
//// Copyright (C) 2021 Intel Corporation
////
//// Licensed under the Apache License, Version 2.0 (the "License");
//// you may not use this file except in compliance with the License.
//// You may obtain a copy of the License at
////
////      http://www.apache.org/licenses/LICENSE-2.0
////
//// Unless required by applicable law or agreed to in writing, software
//// distributed under the License is distributed on an "AS IS" BASIS,
//// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//// See the License for the specific language governing permissions and
//// limitations under the License.
//*/
//
//#include <algorithm>
//#include <string>
//#include <vector>
//
//#include <opencv2/imgproc/imgproc.hpp>
//
//#include "models/associative_embedding_decoder.h"
//#include "models/hpe_model_associative_embedding.h"
//
//#include <utils/common.hpp>
//#include <utils/ocv_common.hpp>
//#include <utils/slog.hpp>
//#include <ngraph/ngraph.hpp>
//
//const cv::Vec3f HpeAssociativeEmbedding::meanPixel = cv::Vec3f::all(128);
//const float HpeAssociativeEmbedding::detectionThreshold = 0.1f;
//const float HpeAssociativeEmbedding::tagThreshold = 1.0f;
//
//HpeAssociativeEmbedding::HpeAssociativeEmbedding(const std::string& modelFileName, double aspectRatio,
//    int targetSize, float confidenceThreshold, float delta, RESIZE_MODE resizeMode) :
//    ModelBase(modelFileName),
//    aspectRatio(aspectRatio),
//    targetSize(targetSize),
//    confidenceThreshold(confidenceThreshold),
//    delta(delta),
//    resizeMode(resizeMode) {
//}
//
//void HpeAssociativeEmbedding::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
//    // --------------------------- Configure input & output -------------------------------------------------
//    // --------------------------- Prepare input blobs ------------------------------------------------------
//    changeInputSize(cnnNetwork);
//
//    InferenceEngine::ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
//    if (inputShapes.size() != 1)
//        throw std::runtime_error("Demo supports topologies only with 1 input");
//    inputsNames.push_back(inputShapes.begin()->first);
//    InferenceEngine::SizeVector& inSizeVector = inputShapes.begin()->second;
//    if (inSizeVector.size() != 4 || inSizeVector[0] != 1 || inSizeVector[1] != 3)
//        throw std::runtime_error("3-channel 4-dimensional model's input is expected");
//    InferenceEngine::InputInfo& inputInfo = *cnnNetwork.getInputsInfo().begin()->second;
//    inputInfo.setPrecision(InferenceEngine::Precision::U8);
//    inputInfo.getInputData()->setLayout(InferenceEngine::Layout::NHWC);
//
//    // --------------------------- Prepare output blobs -----------------------------------------------------
//    const InferenceEngine::OutputsDataMap& outputInfo = cnnNetwork.getOutputsInfo();
//    if (outputInfo.size() != 2 && outputInfo.size() != 3)
//        throw std::runtime_error("Demo supports topologies only with 2 or 3 outputs");
//    for (const auto& outputLayer: outputInfo) {
//        outputLayer.second->setPrecision(InferenceEngine::Precision::FP32);
//        outputsNames.push_back(outputLayer.first);
//        const InferenceEngine::SizeVector& outputLayerDims = outputLayer.second->getTensorDesc().getDims();
//        if (outputLayerDims.size() != 4 && outputLayerDims.size() != 5)
//                throw std::runtime_error("output layers are expected to be 4-dimensional or 5-dimensional");
//        if (outputLayerDims[0] != 1 || outputLayerDims[1] != 17)
//                throw std::runtime_error("output layers are expected to have 1 batch size and 17 channels");
//    }
//    embeddingsBlobName = findLayerByName("embeddings", outputsNames);
//    heatmapsBlobName = findLayerByName("heatmaps", outputsNames);
//    try {
//        nmsHeatmapsBlobName = findLayerByName("nms_heatmaps", outputsNames);
//    }
//    catch (const std::runtime_error&) {
//        nmsHeatmapsBlobName = heatmapsBlobName;
//    }
//}
//
//void HpeAssociativeEmbedding::changeInputSize(InferenceEngine::CNNNetwork& cnnNetwork) {
//    InferenceEngine::ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
//    InferenceEngine::SizeVector& inputDims = inputShapes.begin()->second;
//    if (!targetSize) {
//        targetSize =  static_cast<int>(std::min(inputDims[2], inputDims[3]));
//    }
//    int inputHeight = aspectRatio >= 1.0 ? targetSize : static_cast<int>(std::round(targetSize / aspectRatio));
//    int inputWidth = aspectRatio >= 1.0 ? static_cast<int>(std::round(targetSize * aspectRatio)) : targetSize;
//    int height = static_cast<int>((inputHeight + stride - 1) / stride) * stride;
//    int width = static_cast<int>((inputWidth + stride - 1) / stride) * stride;
//    inputDims[0] = 1;
//    inputDims[2] = height;
//    inputDims[3] = width;
//    inputLayerSize = cv::Size(inputDims[3], inputDims[2]);
//    cnnNetwork.reshape(inputShapes);
//}
//
//std::shared_ptr<InternalModelData> HpeAssociativeEmbedding::preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) {
//    auto& image = inputData.asRef<ImageInputData>().inputImage;
//    cv::Rect roi;
//    auto paddedImage = resizeImageExt(image, inputLayerSize.width, inputLayerSize.height, resizeMode, true, &roi);
//    if (inputLayerSize.height - stride >= roi.height
//        || inputLayerSize.width - stride >= roi.width) {
//        slog::warn << "\tChosen model aspect ratio doesn't match image aspect ratio" << slog::endl;
//    }
//    request->SetBlob(inputsNames[0], wrapMat2Blob(paddedImage));
//
//    return std::make_shared<InternalScaleData>(paddedImage.cols, paddedImage.rows,
//        image.size().width / static_cast<float>(roi.width), image.size().height / static_cast<float>(roi.height));
//}
//
//std::unique_ptr<ResultBase> HpeAssociativeEmbedding::postprocess(InferenceResult& infResult) {
//    HumanPoseResult* result = new HumanPoseResult(infResult.frameId, infResult.metaData);
//
//    auto aembds = infResult.outputsData[embeddingsBlobName];
//    const InferenceEngine::SizeVector& aembdsDims = aembds->getTensorDesc().getDims();
//    float* aembdsMapped = aembds->rmap().as<float*>();
//    std::vector<cv::Mat> aembdsMaps = split(aembdsMapped, aembdsDims);
//
//    auto heats = infResult.outputsData[heatmapsBlobName];
//    const InferenceEngine::SizeVector& heatMapsDims = heats->getTensorDesc().getDims();
//    float* heatMapsMapped = heats->rmap().as<float*>();
//    std::vector<cv::Mat> heatMaps = split(heatMapsMapped, heatMapsDims);
//
//    std::vector<cv::Mat> nmsHeatMaps = heatMaps;
//    if (nmsHeatmapsBlobName != heatmapsBlobName) {
//        auto nmsHeats = infResult.outputsData[nmsHeatmapsBlobName];
//        const InferenceEngine::SizeVector& nmsHeatMapsDims = nmsHeats->getTensorDesc().getDims();
//        float* nmsHeatMapsMapped = nmsHeats->rmap().as<float*>();
//        nmsHeatMaps = split(nmsHeatMapsMapped, nmsHeatMapsDims);
//    }
//    std::vector<HumanPose> poses = extractPoses(heatMaps, aembdsMaps, nmsHeatMaps);
//
//    // Rescale poses to the original image
//    const auto& scale = infResult.internalModelData->asRef<InternalScaleData>();
//    float outputScale = inputLayerSize.width / static_cast<float>(heatMapsDims[3]);
//    float shiftX = 0.0, shiftY = 0.0;
//    float scaleX = 1.0, scaleY = 1.0;
//
//    if (resizeMode == RESIZE_KEEP_ASPECT_LETTERBOX) {
//        scaleX = scaleY = std::min(scale.scaleX, scale.scaleY);
//        if (aspectRatio >= 1.0)
//            shiftX = static_cast<float>((targetSize * scaleX * aspectRatio - scale.inputImgWidth * scaleX) / 2);
//        else
//            shiftY = static_cast<float>((targetSize * scaleY / aspectRatio - scale.inputImgHeight * scaleY) / 2);
//        scaleX = scaleY *= outputScale;
//    } else {
//        scaleX = scale.scaleX * outputScale;
//        scaleY = scale.scaleY * outputScale;
//    }
//
//    for (auto& pose : poses) {
//        for (auto& keypoint : pose.keypoints) {
//            if (keypoint != cv::Point2f(-1, -1)) {
//                keypoint.x = keypoint.x * scaleX + shiftX;
//                keypoint.y = keypoint.y * scaleY + shiftY;
//            }
//        }
//        result->poses.push_back(pose);
//    }
//
//    return std::unique_ptr<ResultBase>(result);
//}
//
//std::string HpeAssociativeEmbedding::findLayerByName(const std::string layerName,
//                                                     const std::vector<std::string>& outputsNames) {
//    std::vector<std::string> suitableLayers;
//    for (auto& outputName: outputsNames) {
//        if (outputName.rfind(layerName, 0) == 0) {
//           suitableLayers.push_back(outputName);
//        }
//    }
//    if (suitableLayers.empty())
//        throw std::runtime_error("Suitable layer for " + layerName + " output is not found");
//    else if (suitableLayers.size() > 1)
//        throw std::runtime_error("More than 1 layer matched to " + layerName + " output");
//    return suitableLayers[0];
//}
//
//std::vector<cv::Mat> HpeAssociativeEmbedding::split(float* data, const InferenceEngine::SizeVector& shape) {
//    std::vector<cv::Mat> flattenData(shape[1]);
//    for (size_t i = 0; i < flattenData.size(); i++) {
//        flattenData[i] = cv::Mat(shape[2], shape[3], CV_32FC1, data + i * shape[2] * shape[3]);
//    }
//    return flattenData;
//}
//
//std::vector<HumanPose> HpeAssociativeEmbedding::extractPoses(
//    std::vector<cv::Mat>& heatMaps,
//    const std::vector<cv::Mat>& aembdsMaps,
//    const std::vector<cv::Mat>& nmsHeatMaps) const {
//
//    std::vector<std::vector<Peak>> allPeaks(numJoints);
//    for (int i = 0; i < numJoints; i++) {
//        findPeaks(nmsHeatMaps, aembdsMaps, allPeaks, i, maxNumPeople, detectionThreshold);
//    }
//    std::vector<Pose> allPoses = matchByTag(allPeaks, maxNumPeople, numJoints, tagThreshold);
//    // swap for all poses
//    for (auto& pose : allPoses) {
//        for (size_t j = 0; j < numJoints; j++) {
//            Peak& peak = pose.getPeak(j);
//            std::swap(peak.keypoint.x, peak.keypoint.y);
//        }
//    }
//    std::vector<HumanPose> poses;
//    for (size_t i = 0; i < allPoses.size(); i++) {
//        Pose& pose = allPoses[i];
//        // Filtering poses with low mean scores
//        if (pose.getMeanScore() <= confidenceThreshold) {
//            continue;
//        }
//        for (size_t j = 0; j < heatMaps.size(); j++) {
//            heatMaps[j] = cv::abs(heatMaps[j]);
//        }
//        adjustAndRefine(allPoses, heatMaps, aembdsMaps, i, delta);
//        std::vector<cv::Point2f> keypoints;
//        for (size_t j = 0; j < numJoints; j++) {
//            Peak& peak = pose.getPeak(j);
//            keypoints.push_back(peak.keypoint);
//        }
//        poses.push_back({keypoints, pose.getMeanScore()});
//    }
//    return poses;
//}
