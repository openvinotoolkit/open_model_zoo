/*
// Copyright (C) 2018-2021 Intel Corporation
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

#include "models/hpe_model_openpose.h"
#include "models/openpose_decoder.h"

#include <utils/common.hpp>
#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>
#include <ngraph/ngraph.hpp>

using namespace InferenceEngine;

const cv::Vec3f HPEOpenPose::meanPixel = cv::Vec3f::all(128);
const float HPEOpenPose::minPeaksDistance = 3.0f;
const float HPEOpenPose::midPointsScoreThreshold = 0.05f;
const float HPEOpenPose::foundMidPointsRatioThreshold = 0.8f;
const float HPEOpenPose::minSubsetScore = 0.2f;

HPEOpenPose::HPEOpenPose(const std::string& modelFileName, double aspectRatio, int targetSize, float confidenceThreshold) :
    ImageModel(modelFileName, false),
    aspectRatio(aspectRatio),
    targetSize(targetSize),
    confidenceThreshold(confidenceThreshold) {
}

void HPEOpenPose::prepareInputsOutputs(CNNNetwork& cnnNetwork) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input blobs ------------------------------------------------------
    changeInputSize(cnnNetwork);

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
    if (outputInfo.size() != 2)
        throw std::runtime_error("Demo supports topologies only with 2 outputs");

    for (const auto& outputLayer: outputInfo) {
        outputLayer.second->setPrecision(Precision::FP32);
        outputLayer.second->setLayout(Layout::NCHW);
        outputsNames.push_back(outputLayer.first);
    }

    auto outputIt = outputInfo.begin();
    const SizeVector& pafsOutputDims = (*outputIt++).second->getTensorDesc().getDims();
    if (pafsOutputDims.size() != 4 || pafsOutputDims[0] != 1 || pafsOutputDims[1] != 2 * (keypointsNumber + 1))
        throw std::runtime_error("1x" + std::to_string(2 * (keypointsNumber + 1)) + "xHFMxWFM dimension of model's output is expected");
    const SizeVector& heatmapsOutputDims = (*outputIt++).second->getTensorDesc().getDims();
    if (heatmapsOutputDims.size() != 4 || heatmapsOutputDims[0] != 1 || heatmapsOutputDims[1] != keypointsNumber + 1)
        throw std::runtime_error("1x" + std::to_string(keypointsNumber + 1) + "xHFMxWFM dimension of model's heatmap is expected");
    if (pafsOutputDims[2] != heatmapsOutputDims[2] || pafsOutputDims[3] != heatmapsOutputDims[3])
        throw std::runtime_error("output and heatmap are expected to have matching last two dimensions");
}

void HPEOpenPose::changeInputSize(CNNNetwork& cnnNetwork) {
    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    SizeVector& inputDims = inputShapes.begin()->second;
    if (!targetSize) {
        targetSize = inputDims[2];
    }
    int height = static_cast<int>((targetSize + stride - 1) / stride) * stride;
    int inputWidth = static_cast<int>(std::round(targetSize * aspectRatio));
    int width = static_cast<int>((inputWidth + stride - 1) / stride) * stride;
    inputDims[0] = 1;
    inputDims[2] = height;
    inputDims[3] = width;
    inputLayerSize = cv::Size(inputDims[3], inputDims[2]);
    cnnNetwork.reshape(inputShapes);
}

std::shared_ptr<InternalModelData> HPEOpenPose::preprocess(const InputData& inputData, InferRequest::Ptr& request) {
    auto& image = inputData.asRef<ImageInputData>().inputImage;
    cv::Mat resizedImage;
    double scale = inputLayerSize.height / static_cast<double>(image.rows);
    cv::resize(image, resizedImage, cv::Size(), scale, scale, cv::INTER_CUBIC);
    int h = resizedImage.rows;
    int w = resizedImage.cols;
    if (inputLayerSize.width < w)
        throw std::runtime_error("The image aspect ratio doesn't fit current model shape");
    if (!(inputLayerSize.width - stride < w && w <= inputLayerSize.width)) {
        slog::warn << "Chosen model aspect ratio doesn't match image aspect ratio\n";
    }
    cv::Mat paddedImage;
    int right = inputLayerSize.width - w;
    cv::copyMakeBorder(resizedImage, paddedImage, 0, 0, 0, right,
                       cv::BORDER_CONSTANT, meanPixel);
    request->SetBlob(inputsNames[0], wrapMat2Blob(paddedImage));
    /* IE::Blob::Ptr from wrapMat2Blob() doesn't own data. Save the image to avoid deallocation before inference */
    return std::make_shared<InternalScaleMatData>(image.cols / static_cast<float>(w), image.rows / static_cast<float>(h), std::move(paddedImage));
}

std::unique_ptr<ResultBase> HPEOpenPose::postprocess(InferenceResult& infResult) {
    HumanPoseResult* result = new HumanPoseResult(infResult.frameId, infResult.metaData);

    auto outputMapped = infResult.outputsData[outputsNames[0]];
    auto heatMapsMapped = infResult.outputsData[outputsNames[1]];

    const SizeVector& outputDims = outputMapped->getTensorDesc().getDims();
    const SizeVector& heatMapDims = heatMapsMapped->getTensorDesc().getDims();

    float* predictions = outputMapped->rmap().as<float*>();
    float* heats = heatMapsMapped->rmap().as<float*>();

    std::vector<cv::Mat> heatMaps(keypointsNumber);
    for (size_t i = 0; i < heatMaps.size(); i++) {
        heatMaps[i] = cv::Mat(heatMapDims[2], heatMapDims[3], CV_32FC1,
                              heats + i * heatMapDims[2] * heatMapDims[3]);
    }
    resizeFeatureMaps(heatMaps);

    std::vector<cv::Mat> pafs(outputDims[1]);
    for (size_t i = 0; i < pafs.size(); i++) {
        pafs[i] = cv::Mat(heatMapDims[2], heatMapDims[3], CV_32FC1,
                          predictions + i * heatMapDims[2] * heatMapDims[3]);
    }
    resizeFeatureMaps(pafs);

    std::vector<HumanPose> poses = extractPoses(heatMaps, pafs);

    const auto& scale = infResult.internalModelData->asRef<InternalScaleMatData>();
    float scaleX = stride / upsampleRatio * scale.x;
    float scaleY = stride / upsampleRatio * scale.y;
    for (auto& pose : poses) {
        for (auto& keypoint : pose.keypoints) {
            if (keypoint != cv::Point2f(-1, -1)) {
                keypoint.x *= scaleX;
                keypoint.y *= scaleY;
            }
        }
    }
    for (size_t i = 0; i < poses.size(); ++i) {
        result->poses.push_back(poses[i]);
    }

    return std::unique_ptr<ResultBase>(result);
}

void HPEOpenPose::resizeFeatureMaps(std::vector<cv::Mat>& featureMaps) const {
    for (auto& featureMap : featureMaps) {
        cv::resize(featureMap, featureMap, cv::Size(),
                   upsampleRatio, upsampleRatio, cv::INTER_CUBIC);
    }
}

class FindPeaksBody: public cv::ParallelLoopBody {
public:
    FindPeaksBody(const std::vector<cv::Mat>& heatMaps, float minPeaksDistance,
                  std::vector<std::vector<Peak> >& peaksFromHeatMap, float confidenceThreshold)
        : heatMaps(heatMaps),
          minPeaksDistance(minPeaksDistance),
          peaksFromHeatMap(peaksFromHeatMap),
          confidenceThreshold(confidenceThreshold) {}

    void operator()(const cv::Range& range) const override {
        for (int i = range.start; i < range.end; i++) {
            findPeaks(heatMaps, minPeaksDistance, peaksFromHeatMap, i, confidenceThreshold);
        }
    }

private:
    const std::vector<cv::Mat>& heatMaps;
    float minPeaksDistance;
    std::vector<std::vector<Peak> >& peaksFromHeatMap;
    float confidenceThreshold;
};

std::vector<HumanPose> HPEOpenPose::extractPoses(
        const std::vector<cv::Mat>& heatMaps,
        const std::vector<cv::Mat>& pafs) const {
    std::vector<std::vector<Peak>> peaksFromHeatMap(heatMaps.size());
    FindPeaksBody findPeaksBody(heatMaps, minPeaksDistance, peaksFromHeatMap, confidenceThreshold);
    cv::parallel_for_(cv::Range(0, static_cast<int>(heatMaps.size())),
                      findPeaksBody);
    int peaksBefore = 0;
    for (size_t heatmapId = 1; heatmapId < heatMaps.size(); heatmapId++) {
        peaksBefore += static_cast<int>(peaksFromHeatMap[heatmapId - 1].size());
        for (auto& peak : peaksFromHeatMap[heatmapId]) {
            peak.id += peaksBefore;
        }
    }
    std::vector<HumanPose> poses = groupPeaksToPoses(
                peaksFromHeatMap, pafs, keypointsNumber, midPointsScoreThreshold,
                foundMidPointsRatioThreshold, minJointsNumber, minSubsetScore);
    return poses;
}
