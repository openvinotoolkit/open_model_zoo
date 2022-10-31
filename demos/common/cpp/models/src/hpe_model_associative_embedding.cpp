/*
// Copyright (C) 2021-2022 Intel Corporation
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

#include "models/hpe_model_associative_embedding.h"

#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>

#include <utils/image_utils.h>
#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>

#include "models/associative_embedding_decoder.h"
#include "models/input_data.h"
#include "models/internal_model_data.h"
#include "models/results.h"

const cv::Vec3f HpeAssociativeEmbedding::meanPixel = cv::Vec3f::all(128);
const float HpeAssociativeEmbedding::detectionThreshold = 0.1f;
const float HpeAssociativeEmbedding::tagThreshold = 1.0f;

HpeAssociativeEmbedding::HpeAssociativeEmbedding(const std::string& modelFileName,
                                                 double aspectRatio,
                                                 int targetSize,
                                                 float confidenceThreshold,
                                                 const std::string& layout,
                                                 float delta,
                                                 RESIZE_MODE resizeMode)
    : ImageModel(modelFileName, false, layout),
      aspectRatio(aspectRatio),
      targetSize(targetSize),
      confidenceThreshold(confidenceThreshold),
      delta(delta),
      resizeMode(resizeMode) {}

void HpeAssociativeEmbedding::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input Tensors ------------------------------------------------------
    if (model->inputs().size() != 1) {
        throw std::logic_error("HPE AE model wrapper supports topologies with only 1 input.");
    }
    inputsNames.push_back(model->input().get_any_name());

    const ov::Shape& inputShape = model->input().get_shape();
    const ov::Layout& inputLayout = getInputLayout(model->input());

    if (inputShape.size() != 4 || inputShape[ov::layout::batch_idx(inputLayout)] != 1 ||
        inputShape[ov::layout::channels_idx(inputLayout)] != 3) {
        throw std::logic_error("3-channel 4-dimensional model's input is expected");
    }

    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout({"NHWC"});

    ppp.input().model().set_layout(inputLayout);

    // --------------------------- Prepare output Tensors -----------------------------------------------------
    const ov::OutputVector& outputs = model->outputs();
    if (outputs.size() != 2 && outputs.size() != 3) {
        throw std::logic_error("HPE AE model model wrapper supports topologies only with 2 or 3 outputs");
    }

    for (const auto& output : model->outputs()) {
        const auto& outTensorName = output.get_any_name();
        ppp.output(outTensorName).tensor().set_element_type(ov::element::f32);

        for (const auto& name : output.get_names()) {
            outputsNames.push_back(name);
        }

        const ov::Shape& outputShape = output.get_shape();
        if (outputShape.size() != 4 && outputShape.size() != 5) {
            throw std::logic_error("output tensors are expected to be 4-dimensional or 5-dimensional");
        }
        if (outputShape[ov::layout::batch_idx("NC...")] != 1 || outputShape[ov::layout::channels_idx("NC...")] != 17) {
            throw std::logic_error("output tensors are expected to have 1 batch size and 17 channels");
        }
    }
    model = ppp.build();

    embeddingsTensorName = findTensorByName("embeddings", outputsNames);
    heatmapsTensorName = findTensorByName("heatmaps", outputsNames);
    try {
        nmsHeatmapsTensorName = findTensorByName("nms_heatmaps", outputsNames);
    } catch (const std::runtime_error&) { nmsHeatmapsTensorName = heatmapsTensorName; }

    changeInputSize(model);
}

void HpeAssociativeEmbedding::changeInputSize(std::shared_ptr<ov::Model>& model) {
    ov::Shape inputShape = model->input().get_shape();
    const ov::Layout& layout = ov::layout::get_layout(model->input());
    const auto batchId = ov::layout::batch_idx(layout);
    const auto heightId = ov::layout::height_idx(layout);
    const auto widthId = ov::layout::width_idx(layout);

    if (!targetSize) {
        targetSize = static_cast<int>(std::min(inputShape[heightId], inputShape[widthId]));
    }
    int inputHeight = aspectRatio >= 1.0 ? targetSize : static_cast<int>(std::round(targetSize / aspectRatio));
    int inputWidth = aspectRatio >= 1.0 ? static_cast<int>(std::round(targetSize * aspectRatio)) : targetSize;
    int height = static_cast<int>((inputHeight + stride - 1) / stride) * stride;
    int width = static_cast<int>((inputWidth + stride - 1) / stride) * stride;
    inputShape[batchId] = 1;
    inputShape[heightId] = height;
    inputShape[widthId] = width;
    inputLayerSize = cv::Size(width, height);

    model->reshape(inputShape);
}

std::shared_ptr<InternalModelData> HpeAssociativeEmbedding::preprocess(const InputData& inputData,
                                                                       ov::InferRequest& request) {
    auto& image = inputData.asRef<ImageInputData>().inputImage;
    cv::Rect roi;
    auto paddedImage = resizeImageExt(image, inputLayerSize.width, inputLayerSize.height, resizeMode, true, &roi);
    if (inputLayerSize.height - stride >= roi.height || inputLayerSize.width - stride >= roi.width) {
        slog::warn << "\tChosen model aspect ratio doesn't match image aspect ratio" << slog::endl;
    }
    request.set_input_tensor(wrapMat2Tensor(paddedImage));

    return std::make_shared<InternalScaleData>(paddedImage.cols,
                                               paddedImage.rows,
                                               image.size().width / static_cast<float>(roi.width),
                                               image.size().height / static_cast<float>(roi.height));
}

std::unique_ptr<ResultBase> HpeAssociativeEmbedding::postprocess(InferenceResult& infResult) {
    HumanPoseResult* result = new HumanPoseResult(infResult.frameId, infResult.metaData);

    const auto& aembds = infResult.outputsData[embeddingsTensorName];
    const ov::Shape& aembdsShape = aembds.get_shape();
    float* const aembdsMapped = aembds.data<float>();
    std::vector<cv::Mat> aembdsMaps = split(aembdsMapped, aembdsShape);

    const auto& heats = infResult.outputsData[heatmapsTensorName];
    const ov::Shape& heatMapsShape = heats.get_shape();
    float* const heatMapsMapped = heats.data<float>();
    std::vector<cv::Mat> heatMaps = split(heatMapsMapped, heatMapsShape);

    std::vector<cv::Mat> nmsHeatMaps = heatMaps;
    if (nmsHeatmapsTensorName != heatmapsTensorName) {
        const auto& nmsHeats = infResult.outputsData[nmsHeatmapsTensorName];
        const ov::Shape& nmsHeatMapsShape = nmsHeats.get_shape();
        float* const nmsHeatMapsMapped = nmsHeats.data<float>();
        nmsHeatMaps = split(nmsHeatMapsMapped, nmsHeatMapsShape);
    }
    std::vector<HumanPose> poses = extractPoses(heatMaps, aembdsMaps, nmsHeatMaps);

    // Rescale poses to the original image
    const auto& scale = infResult.internalModelData->asRef<InternalScaleData>();
    const float outputScale = inputLayerSize.width / static_cast<float>(heatMapsShape[3]);
    float shiftX = 0.0, shiftY = 0.0;
    float scaleX = 1.0, scaleY = 1.0;

    if (resizeMode == RESIZE_KEEP_ASPECT_LETTERBOX) {
        scaleX = scaleY = std::min(scale.scaleX, scale.scaleY);
        if (aspectRatio >= 1.0)
            shiftX = static_cast<float>((targetSize * scaleX * aspectRatio - scale.inputImgWidth * scaleX) / 2);
        else
            shiftY = static_cast<float>((targetSize * scaleY / aspectRatio - scale.inputImgHeight * scaleY) / 2);
        scaleX = scaleY *= outputScale;
    } else {
        scaleX = scale.scaleX * outputScale;
        scaleY = scale.scaleY * outputScale;
    }

    for (auto& pose : poses) {
        for (auto& keypoint : pose.keypoints) {
            if (keypoint != cv::Point2f(-1, -1)) {
                keypoint.x = keypoint.x * scaleX + shiftX;
                keypoint.y = keypoint.y * scaleY + shiftY;
            }
        }
        result->poses.push_back(pose);
    }

    return std::unique_ptr<ResultBase>(result);
}

std::string HpeAssociativeEmbedding::findTensorByName(const std::string& tensorName,
                                                      const std::vector<std::string>& outputsNames) {
    std::vector<std::string> suitableLayers;
    for (auto& outputName : outputsNames) {
        if (outputName.rfind(tensorName, 0) == 0) {
            suitableLayers.push_back(outputName);
        }
    }
    if (suitableLayers.empty()) {
        throw std::runtime_error("Suitable tensor for " + tensorName + " output is not found");
    } else if (suitableLayers.size() > 1) {
        throw std::runtime_error("More than 1 tensor matched to " + tensorName + " output");
    }
    return suitableLayers[0];
}

std::vector<cv::Mat> HpeAssociativeEmbedding::split(float* data, const ov::Shape& shape) {
    std::vector<cv::Mat> flattenData(shape[1]);
    for (size_t i = 0; i < flattenData.size(); i++) {
        flattenData[i] = cv::Mat(shape[2], shape[3], CV_32FC1, data + i * shape[2] * shape[3]);
    }
    return flattenData;
}

std::vector<HumanPose> HpeAssociativeEmbedding::extractPoses(std::vector<cv::Mat>& heatMaps,
                                                             const std::vector<cv::Mat>& aembdsMaps,
                                                             const std::vector<cv::Mat>& nmsHeatMaps) const {
    std::vector<std::vector<Peak>> allPeaks(numJoints);
    for (int i = 0; i < numJoints; i++) {
        findPeaks(nmsHeatMaps, aembdsMaps, allPeaks, i, maxNumPeople, detectionThreshold);
    }
    std::vector<Pose> allPoses = matchByTag(allPeaks, maxNumPeople, numJoints, tagThreshold);
    // swap for all poses
    for (auto& pose : allPoses) {
        for (size_t j = 0; j < numJoints; j++) {
            Peak& peak = pose.getPeak(j);
            std::swap(peak.keypoint.x, peak.keypoint.y);
        }
    }
    std::vector<HumanPose> poses;
    for (size_t i = 0; i < allPoses.size(); i++) {
        Pose& pose = allPoses[i];
        // Filtering poses with low mean scores
        if (pose.getMeanScore() <= confidenceThreshold) {
            continue;
        }
        for (size_t j = 0; j < heatMaps.size(); j++) {
            heatMaps[j] = cv::abs(heatMaps[j]);
        }
        adjustAndRefine(allPoses, heatMaps, aembdsMaps, i, delta);
        std::vector<cv::Point2f> keypoints;
        for (size_t j = 0; j < numJoints; j++) {
            Peak& peak = pose.getPeak(j);
            keypoints.push_back(peak.keypoint);
        }
        poses.push_back({keypoints, pose.getMeanScore()});
    }
    return poses;
}
