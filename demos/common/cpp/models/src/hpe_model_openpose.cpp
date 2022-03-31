/*
// Copyright (C) 2020-2022 Intel Corporation
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

#include "models/hpe_model_openpose.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include <utils/image_utils.h>
#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>

#include "models/input_data.h"
#include "models/internal_model_data.h"
#include "models/openpose_decoder.h"
#include "models/results.h"

const cv::Vec3f HPEOpenPose::meanPixel = cv::Vec3f::all(128);
const float HPEOpenPose::minPeaksDistance = 3.0f;
const float HPEOpenPose::midPointsScoreThreshold = 0.05f;
const float HPEOpenPose::foundMidPointsRatioThreshold = 0.8f;
const float HPEOpenPose::minSubsetScore = 0.2f;

HPEOpenPose::HPEOpenPose(const std::string& modelFileName,
                         double aspectRatio,
                         int targetSize,
                         float confidenceThreshold,
                         const std::string& layout)
    : ImageModel(modelFileName, false, layout),
      aspectRatio(aspectRatio),
      targetSize(targetSize),
      confidenceThreshold(confidenceThreshold) {}

void HPEOpenPose::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input  ------------------------------------------------------
    if (model->inputs().size() != 1) {
        throw std::logic_error("HPE OpenPose model wrapper supports topologies with only 1 input");
    }
    inputsNames.push_back(model->input().get_any_name());
    const ov::Shape& inputShape = model->input().get_shape();
    const ov::Layout& inputLayout = getInputLayout(model->input());

    if (inputShape.size() != 4 || inputShape[ov::layout::batch_idx(inputLayout)] != 1 ||
        inputShape[ov::layout::channels_idx(inputLayout)] != 3)
        throw std::logic_error("3-channel 4-dimensional model's input is expected");

    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout({"NHWC"});

    ppp.input().model().set_layout(inputLayout);

    // --------------------------- Prepare output  -----------------------------------------------------
    const ov::OutputVector& outputs = model->outputs();
    if (outputs.size() != 2) {
        throw std::runtime_error("HPE OpenPose supports topologies with only 2 outputs");
    }

    const ov::Layout outputLayout("NCHW");
    for (const auto& output : model->outputs()) {
        const auto& outTensorName = output.get_any_name();
        ppp.output(outTensorName).tensor().set_element_type(ov::element::f32).set_layout(outputLayout);
        outputsNames.push_back(outTensorName);
    }
    model = ppp.build();

    const size_t batchId = ov::layout::batch_idx(outputLayout);
    const size_t channelsId = ov::layout::channels_idx(outputLayout);
    const size_t widthId = ov::layout::width_idx(outputLayout);
    const size_t heightId = ov::layout::height_idx(outputLayout);

    ov::Shape heatmapsOutputShape = model->outputs().front().get_shape();
    ov::Shape pafsOutputShape = model->outputs().back().get_shape();
    if (heatmapsOutputShape[channelsId] > pafsOutputShape[channelsId]) {
        std::swap(heatmapsOutputShape, pafsOutputShape);
        std::swap(outputsNames[0], outputsNames[1]);
    }

    if (heatmapsOutputShape.size() != 4 || heatmapsOutputShape[batchId] != 1 ||
        heatmapsOutputShape[ov::layout::channels_idx(outputLayout)] != keypointsNumber + 1) {
        throw std::logic_error("1x" + std::to_string(keypointsNumber + 1) +
                               "xHFMxWFM dimension of model's heatmap is expected");
    }
    if (pafsOutputShape.size() != 4 || pafsOutputShape[batchId] != 1 ||
        pafsOutputShape[channelsId] != 2 * (keypointsNumber + 1)) {
        throw std::logic_error("1x" + std::to_string(2 * (keypointsNumber + 1)) +
                               "xHFMxWFM dimension of model's output is expected");
    }
    if (pafsOutputShape[heightId] != heatmapsOutputShape[heightId] ||
        pafsOutputShape[widthId] != heatmapsOutputShape[widthId]) {
        throw std::logic_error("output and heatmap are expected to have matching last two dimensions");
    }

    changeInputSize(model);
}

void HPEOpenPose::changeInputSize(std::shared_ptr<ov::Model>& model) {
    ov::Shape inputShape = model->input().get_shape();
    const ov::Layout& layout = ov::layout::get_layout(model->inputs().front());
    const auto batchId = ov::layout::batch_idx(layout);
    const auto heightId = ov::layout::height_idx(layout);
    const auto widthId = ov::layout::width_idx(layout);

    if (!targetSize) {
        targetSize = inputShape[heightId];
    }
    int height = static_cast<int>((targetSize + stride - 1) / stride) * stride;
    int inputWidth = static_cast<int>(std::round(targetSize * aspectRatio));
    int width = static_cast<int>((inputWidth + stride - 1) / stride) * stride;
    inputShape[batchId] = 1;
    inputShape[heightId] = height;
    inputShape[widthId] = width;
    inputLayerSize = cv::Size(width, height);
    model->reshape(inputShape);
}

std::shared_ptr<InternalModelData> HPEOpenPose::preprocess(const InputData& inputData, ov::InferRequest& request) {
    auto& image = inputData.asRef<ImageInputData>().inputImage;
    cv::Rect roi;
    auto paddedImage =
        resizeImageExt(image, inputLayerSize.width, inputLayerSize.height, RESIZE_KEEP_ASPECT, true, &roi);
    if (inputLayerSize.width < roi.width)
        throw std::runtime_error("The image aspect ratio doesn't fit current model shape");

    if (inputLayerSize.width - stride >= roi.width) {
        slog::warn << "\tChosen model aspect ratio doesn't match image aspect ratio" << slog::endl;
    }

    request.set_input_tensor(wrapMat2Tensor(paddedImage));
    return std::make_shared<InternalScaleData>(paddedImage.cols,
                                               paddedImage.rows,
                                               image.cols / static_cast<float>(roi.width),
                                               image.rows / static_cast<float>(roi.height));
}

std::unique_ptr<ResultBase> HPEOpenPose::postprocess(InferenceResult& infResult) {
    HumanPoseResult* result = new HumanPoseResult(infResult.frameId, infResult.metaData);

    const auto& heatMapsMapped = infResult.outputsData[outputsNames[0]];
    const auto& outputMapped = infResult.outputsData[outputsNames[1]];

    const ov::Shape& outputShape = outputMapped.get_shape();
    const ov::Shape& heatMapShape = heatMapsMapped.get_shape();

    float* const predictions = outputMapped.data<float>();
    float* const heats = heatMapsMapped.data<float>();

    std::vector<cv::Mat> heatMaps(keypointsNumber);
    for (size_t i = 0; i < heatMaps.size(); i++) {
        heatMaps[i] =
            cv::Mat(heatMapShape[2], heatMapShape[3], CV_32FC1, heats + i * heatMapShape[2] * heatMapShape[3]);
    }
    resizeFeatureMaps(heatMaps);

    std::vector<cv::Mat> pafs(outputShape[1]);
    for (size_t i = 0; i < pafs.size(); i++) {
        pafs[i] =
            cv::Mat(heatMapShape[2], heatMapShape[3], CV_32FC1, predictions + i * heatMapShape[2] * heatMapShape[3]);
    }
    resizeFeatureMaps(pafs);

    std::vector<HumanPose> poses = extractPoses(heatMaps, pafs);

    const auto& scale = infResult.internalModelData->asRef<InternalScaleData>();
    float scaleX = stride / upsampleRatio * scale.scaleX;
    float scaleY = stride / upsampleRatio * scale.scaleY;
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
        cv::resize(featureMap, featureMap, cv::Size(), upsampleRatio, upsampleRatio, cv::INTER_CUBIC);
    }
}

class FindPeaksBody : public cv::ParallelLoopBody {
public:
    FindPeaksBody(const std::vector<cv::Mat>& heatMaps,
                  float minPeaksDistance,
                  std::vector<std::vector<Peak>>& peaksFromHeatMap,
                  float confidenceThreshold)
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
    std::vector<std::vector<Peak>>& peaksFromHeatMap;
    float confidenceThreshold;
};

std::vector<HumanPose> HPEOpenPose::extractPoses(const std::vector<cv::Mat>& heatMaps,
                                                 const std::vector<cv::Mat>& pafs) const {
    std::vector<std::vector<Peak>> peaksFromHeatMap(heatMaps.size());
    FindPeaksBody findPeaksBody(heatMaps, minPeaksDistance, peaksFromHeatMap, confidenceThreshold);
    cv::parallel_for_(cv::Range(0, static_cast<int>(heatMaps.size())), findPeaksBody);
    int peaksBefore = 0;
    for (size_t heatmapId = 1; heatmapId < heatMaps.size(); heatmapId++) {
        peaksBefore += static_cast<int>(peaksFromHeatMap[heatmapId - 1].size());
        for (auto& peak : peaksFromHeatMap[heatmapId]) {
            peak.id += peaksBefore;
        }
    }
    std::vector<HumanPose> poses = groupPeaksToPoses(peaksFromHeatMap,
                                                     pafs,
                                                     keypointsNumber,
                                                     midPointsScoreThreshold,
                                                     foundMidPointsRatioThreshold,
                                                     minJointsNumber,
                                                     minSubsetScore);
    return poses;
}
