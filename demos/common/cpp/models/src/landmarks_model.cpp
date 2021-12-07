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

#include "models/landmarks_model.h"
#include <utils/common.hpp>
#include <algorithm>
#include <vector>
#include <utility>

LandmarksModel::LandmarksModel(const std::string& modelFileName, bool useAutoResize, std::string postprocessKey) :
    ImageModel(modelFileName, useAutoResize) {
    postprocessType = postprocessKey;
}


void LandmarksModel::prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input blobs ------------------------------------------------------
    InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    if (inputInfo.size() != 1) {
        throw std::logic_error("Landmarks network should have only one input");
    }
    inputsNames.push_back(inputInfo.begin()->first);
    auto layerData = inputInfo.begin()->second;
    auto layerDims = layerData->getTensorDesc().getDims();
    const InferenceEngine::TensorDesc& inputDesc = layerData->getTensorDesc();
    netInputHeight = getTensorHeight(inputDesc);
    netInputWidth = getTensorWidth(inputDesc);

    if (layerDims.size() == 4) {
        layerData->setLayout(InferenceEngine::Layout::NCHW);
        layerData->setPrecision(InferenceEngine::Precision::U8);
    }
    else if (layerDims.size() == 2) {
        layerData->setLayout(InferenceEngine::Layout::NC);
        layerData->setPrecision(InferenceEngine::Precision::FP32);
    }
    else {
        throw std::runtime_error("Unknown type of input layer layout. Expected either 4 or 2 dimensional inputs");
    }
    // --------------------------- Prepare output blobs -----------------------------------------------------
    InferenceEngine::OutputsDataMap outputInfo(cnnNetwork.getOutputsInfo());
    outputsNames.push_back(outputInfo.begin()->first);

    if (outputInfo.size() != 1) {
        throw std::logic_error("Landmarks network should have only one output");
    }
    InferenceEngine::Data& output = *outputInfo.begin()->second;
    output.setPrecision(InferenceEngine::Precision::FP32);
    const InferenceEngine::SizeVector& outSizeVector = output.getTensorDesc().getDims();
    if (outSizeVector.size() != 2 && outSizeVector.size() != 4) {
        throw std::logic_error("Landmarks Estimation network output layer should have 2 or 4 dimensions");
    }
}


std::shared_ptr<InternalModelData> LandmarksModel::preprocess(const InputData& inputData, InferenceEngine::InferRequest::Ptr& request) {
    const auto& origImg = inputData.asRef<ImageInputData>().inputImage;
    const auto& img = inputTransform(origImg);
    cv::Mat resizedImage;
    auto scaledSize = cv::Size(static_cast<int>(netInputWidth), static_cast<int>(netInputHeight));
    cv::resize(img, resizedImage, scaledSize, 0, 0, cv::INTER_CUBIC);
    auto inputBlob = request->GetBlob(inputsNames[0]);
    matToBlob(resizedImage, inputBlob);
    return std::make_shared<InternalImageModelData>(img.cols, img.rows);
}

std::unique_ptr<ResultBase> LandmarksModel::postprocess(InferenceResult& infResult) {
    if (postprocessType == "simple") {
        return simplePostprocess(infResult);
    }
    else if(postprocessType == "heatmap"){
        return heatmapPostprocess(infResult);
    }
    else {
        throw std::logic_error("postprocessType parameter is incorrect");
    }
}

std::unique_ptr<ResultBase> LandmarksModel::simplePostprocess(InferenceResult& infResult) {
    InferenceEngine::LockedMemory<const void> outputMapped = infResult.getFirstOutputBlob()->rmap();
    InferenceEngine::MemoryBlob::Ptr  output = infResult.getFirstOutputBlob();
    numberLandmarks = output->getTensorDesc().getDims()[1];
    auto normed_coordinates = output->rmap().as<float*>();
    const auto& internalData = infResult.internalModelData->asRef<InternalImageModelData>();
    LandmarksResult* result = new LandmarksResult(infResult.frameId, infResult.metaData);
    auto retVal = std::unique_ptr<ResultBase>(result);
    for (auto i = 0; i < numberLandmarks / 2; ++i) {
        int normed_x = static_cast<int>(normed_coordinates[2 * i] * internalData.inputImgHeight);
        int normed_y = static_cast<int>(normed_coordinates[2 * i + 1] * internalData.inputImgWidth);

        result->coordinates.push_back(cv::Point2f(normed_x, normed_y));
    }
    return retVal;
}

std::unique_ptr<ResultBase> LandmarksModel::heatmapPostprocess(InferenceResult& infResult) {
    InferenceEngine::MemoryBlob::Ptr  outputMapped = infResult.getFirstOutputBlob();
    const InferenceEngine::SizeVector& heatMapsDims = outputMapped->getTensorDesc().getDims();
    const auto& internalData = infResult.internalModelData->asRef<InternalImageModelData>();
    numberLandmarks = heatMapsDims[1];
    auto data = outputMapped->rmap().as<float*>();
    std::vector<cv::Mat> heatMaps = split(data, heatMapsDims);
    for (size_t i = 0; i < numberLandmarks; ++i) {
        for (size_t j = heatMaps[i].cols - 1; j > 0; --j) {
            cv::Mat tmpCol= heatMaps[i].col(j);
            heatMaps[i].col(j - 1).copyTo(tmpCol);
        }
    }
    cv::Point2f center(internalData.inputImgWidth *0.5, internalData.inputImgHeight *0.5);
    cv::Point2f scale(internalData.inputImgWidth, internalData.inputImgHeight);
    std::vector<cv::Point2f> preds = getMaxPreds(heatMaps);
    for (size_t landmarkId = 0; landmarkId < numberLandmarks; landmarkId++) {
        const cv::Mat& heatMap = heatMaps[landmarkId];
        int px = int(preds[landmarkId].x);
        int py = int(preds[landmarkId].y);
        if (1 < px && px < heatMap.cols - 1 && 1 < py && py < heatMap.rows - 1) {
            float diffFirst = heatMap.at<float>(py, px + 1) - heatMap.at<float>(py, px - 1);
            float diffSecond = heatMap.at<float>(py + 1, px ) - heatMap.at<float>(py - 1, px);
            preds[landmarkId].x += sign(diffFirst) * 0.25;
            preds[landmarkId].y += sign(diffSecond) * 0.25;
        }
    }
    //transform preds
    cv::Mat trans = affineTransform(center, scale, 0, heatMapsDims[2], heatMapsDims[3], cv::Point2f(0., 0.), true);
    std::vector<cv::Point2f> landmarks;
    for (size_t landmarkId = 0; landmarkId < numberLandmarks; landmarkId++) {
        cv::Mat coord(3, 1, CV_32F);
        coord.at<float>(0, 0) = preds[landmarkId].x;
        coord.at<float>(1, 0) = preds[landmarkId].y;
        coord.at<float>(2, 0) = 1;
        cv::Mat point;
        trans.convertTo(trans, CV_32F);
        point = trans * coord;
        landmarks.push_back(cv::Point2f(point.at<float>(0, 0), point.at<float>(1, 0)));
    }
    LandmarksResult* result = new LandmarksResult(infResult.frameId, infResult.metaData);
    auto retVal = std::unique_ptr<ResultBase>(result);
    result->coordinates = landmarks;
    return retVal;
}

std::vector<cv::Mat> LandmarksModel::split(float* data, const InferenceEngine::SizeVector& shape) {
    std::vector<cv::Mat> flattenData(shape[1]);
    for (size_t i = 0; i < flattenData.size(); i++) {
        flattenData[i] = cv::Mat(shape[2], shape[3], CV_32FC1, data + i * shape[2] * shape[3]);
    }
    return flattenData;
}

std::vector<cv::Point2f> LandmarksModel::getMaxPreds(std::vector<cv::Mat> heatMaps) {
    std::vector<cv::Point2f> preds;
    size_t reshapedSize = heatMaps[0].cols * heatMaps[0].rows;
    for (size_t landmarkId = 0; landmarkId < numberLandmarks; landmarkId++) {
        const cv::Mat& heatMap = heatMaps[landmarkId];
        const float* heatMapData = heatMap.ptr<float>();
        std::vector<int> indices(reshapedSize);
        std::iota(std::begin(indices), std::end(indices), 0);
        std::partial_sort(std::begin(indices), std::begin(indices) + numberLandmarks, std::end(indices),
            [heatMapData](int l, int r) { return heatMapData[l] > heatMapData[r]; });
        size_t idx = indices[0];
        float maxVal = heatMapData[idx];
        if (maxVal > 0) {
            preds.push_back(cv::Point2f(idx % heatMaps[0].cols, idx / heatMaps[0].cols));
        }
        else {
            preds.push_back(cv::Point2f(-1, -1));
        }
    }
    return preds;
}

int LandmarksModel::sign(float number) {
    if (number > 0) {
        return 1;
    }
    else if (number < 0) {
        return -1;
    }
    return 0;
}

cv::Mat LandmarksModel::affineTransform(cv::Point2f center, cv::Point2f scale,
    float rot, size_t dst_w, size_t dst_h, cv::Point2f shift, bool inv) {
    cv::Point2f scale_tmp = scale;
    const float pi = acos(-1.0);
    float rot_rad = pi * rot / 180;
    cv::Point2f src_dir = rotatePoint(cv::Point2f(0., scale_tmp.y * -0.5), rot_rad);
    cv::Point2f* src = new cv::Point2f[3];
    src[0] = cv::Point2f(center.x + scale_tmp.x * shift.x, center.y + scale_tmp.y * shift.y);
    src[1] = cv::Point2f(center.x + src_dir.x+ scale_tmp.x * shift.x, center.y + src_dir.y + scale_tmp.y * shift.y);
    src[2] = get3rdPoint(src[0], src[1]);
    cv::Point2f dst_dir = cv::Point2f(0., dst_w * -0.5);
    cv::Point2f* dst = new cv::Point2f[3];
    dst[0] = cv::Point2f(dst_w * 0.5, dst_h * 0.5);
    dst[1] = dst[0]+ dst_dir;
    dst[2] = get3rdPoint(dst[0], dst[1]);
    cv::Mat trans;
    if (inv) {
        trans = cv::getAffineTransform(dst,src);
    }
    else {
        trans = cv::getAffineTransform(src, dst);
    }
    return trans;
}
cv::Point2f LandmarksModel::rotatePoint(cv::Point2f pt, float angle_rad) {
    float new_x = pt.x * cos(angle_rad) - pt.y * sin(angle_rad);
    float new_y = pt.x * sin(angle_rad) + pt.y * cos(angle_rad);
    return  cv::Point2f(new_x, new_y);
}
cv::Point2f LandmarksModel::get3rdPoint(cv::Point2f a, cv::Point2f b) {
    cv::Point2f direction = a - b;
    return  cv::Point2f(b.x - direction.y, b.y + direction.x);
}
