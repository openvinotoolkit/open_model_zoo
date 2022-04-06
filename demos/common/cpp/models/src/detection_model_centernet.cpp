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

#include "models/detection_model_centernet.h"

#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include <utils/common.hpp>
#include <utils/image_utils.h>
#include <utils/ocv_common.hpp>

#include "models/input_data.h"
#include "models/internal_model_data.h"
#include "models/results.h"

ModelCenterNet::ModelCenterNet(const std::string& modelFileName,
                               float confidenceThreshold,
                               const std::vector<std::string>& labels,
                               const std::string& layout)
    : DetectionModel(modelFileName, confidenceThreshold, false, labels, layout) {}

void ModelCenterNet::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input  ------------------------------------------------------
    if (model->inputs().size() != 1) {
        throw std::logic_error("CenterNet model wrapper expects models that have only 1 input");
    }

    const ov::Shape& inputShape = model->input().get_shape();
    const ov::Layout& inputLayout = getInputLayout(model->input());

    if (inputShape[ov::layout::channels_idx(inputLayout)] != 3) {
        throw std::logic_error("Expected 3-channel input");
    }

    ov::preprocess::PrePostProcessor ppp(model);
    inputTransform.setPrecision(ppp, model->input().get_any_name());
    ppp.input().tensor().set_layout("NHWC");

    ppp.input().model().set_layout(inputLayout);

    // --------------------------- Reading image input parameters -------------------------------------------
    inputsNames.push_back(model->input().get_any_name());
    netInputWidth = inputShape[ov::layout::width_idx(inputLayout)];
    netInputHeight = inputShape[ov::layout::height_idx(inputLayout)];

    // --------------------------- Prepare output  -----------------------------------------------------
    if (model->outputs().size() != 3) {
        throw std::logic_error("CenterNet model wrapper expects models that have 3 outputs");
    }

    const ov::Layout outLayout{"NCHW"};
    for (const auto& output : model->outputs()) {
        auto outTensorName = output.get_any_name();
        outputsNames.push_back(outTensorName);
        ppp.output(outTensorName).tensor().set_element_type(ov::element::f32).set_layout(outLayout);
    }
    std::sort(outputsNames.begin(), outputsNames.end());
    model = ppp.build();
}

cv::Point2f getDir(const cv::Point2f& srcPoint, float rotRadius) {
    float sn = sinf(rotRadius);
    float cs = cosf(rotRadius);

    cv::Point2f srcResult(0.0f, 0.0f);
    srcResult.x = srcPoint.x * cs - srcPoint.y * sn;
    srcResult.y = srcPoint.x * sn + srcPoint.y * cs;

    return srcResult;
}

cv::Point2f get3rdPoint(const cv::Point2f& a, const cv::Point2f& b) {
    cv::Point2f direct = a - b;
    return b + cv::Point2f(-direct.y, direct.x);
}

cv::Mat getAffineTransform(float centerX,
                           float centerY,
                           int srcW,
                           float rot,
                           size_t outputWidth,
                           size_t outputHeight,
                           bool inv = false) {
    float rotRad = static_cast<float>(CV_PI) * rot / 180.0f;
    auto srcDir = getDir({0.0f, -0.5f * srcW}, rotRad);
    cv::Point2f dstDir(0.0f, -0.5f * outputWidth);
    std::vector<cv::Point2f> src(3, {0.0f, 0.0f});
    std::vector<cv::Point2f> dst(3, {0.0f, 0.0f});

    src[0] = {centerX, centerY};
    src[1] = srcDir + src[0];
    src[2] = get3rdPoint(src[0], src[1]);

    dst[0] = {outputWidth * 0.5f, outputHeight * 0.5f};
    dst[1] = dst[0] + dstDir;
    dst[2] = get3rdPoint(dst[0], dst[1]);

    cv::Mat trans;
    if (inv) {
        trans = cv::getAffineTransform(dst, src);
    } else {
        trans = cv::getAffineTransform(src, dst);
    }

    return trans;
}

std::shared_ptr<InternalModelData> ModelCenterNet::preprocess(const InputData& inputData, ov::InferRequest& request) {
    auto& img = inputData.asRef<ImageInputData>().inputImage;
    const auto& resizedImg = resizeImageExt(img, netInputWidth, netInputHeight, RESIZE_KEEP_ASPECT_LETTERBOX);

    request.set_input_tensor(wrapMat2Tensor(inputTransform(resizedImg)));
    return std::make_shared<InternalImageModelData>(img.cols, img.rows);
}

std::vector<std::pair<size_t, float>> nms(float* scoresPtr, const ov::Shape& shape, float threshold, int kernel = 3) {
    std::vector<std::pair<size_t, float>> scores;
    scores.reserve(ModelCenterNet::INIT_VECTOR_SIZE);
    auto chSize = shape[2] * shape[3];

    for (size_t i = 0; i < shape[1] * shape[2] * shape[3]; ++i) {
        scoresPtr[i] = expf(scoresPtr[i]) / (1 + expf(scoresPtr[i]));
    }

    for (size_t ch = 0; ch < shape[1]; ++ch) {
        for (size_t w = 0; w < shape[2]; ++w) {
            for (size_t h = 0; h < shape[3]; ++h) {
                float max = scoresPtr[chSize * ch + shape[2] * w + h];

                // ---------------------  filter on threshold--------------------------------------
                if (max < threshold) {
                    continue;
                }

                // ---------------------  store index and score------------------------------------
                scores.push_back({chSize * ch + shape[2] * w + h, max});

                bool next = true;
                // ---------------------- maxpool2d -----------------------------------------------
                for (int i = -kernel / 2; i < kernel / 2 + 1 && next; ++i) {
                    for (int j = -kernel / 2; j < kernel / 2 + 1; ++j) {
                        if (w + i >= 0 && w + i < shape[2] && h + j >= 0 && h + j < shape[3]) {
                            if (scoresPtr[chSize * ch + shape[2] * (w + i) + h + j] > max) {
                                scores.pop_back();
                                next = false;
                                break;
                            }
                        } else {
                            if (max < 0) {
                                scores.pop_back();
                                next = false;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    return scores;
}

static std::vector<std::pair<size_t, float>> filterScores(const ov::Tensor& scoresTensor, float threshold) {
    auto shape = scoresTensor.get_shape();
    float* scoresPtr = scoresTensor.data<float>();

    return nms(scoresPtr, shape, threshold);
}

std::vector<std::pair<float, float>> filterReg(const ov::Tensor& regressionTensor,
                                               const std::vector<std::pair<size_t, float>>& scores,
                                               size_t chSize) {
    const float* regPtr = regressionTensor.data<float>();
    std::vector<std::pair<float, float>> reg;

    for (auto s : scores) {
        reg.push_back({regPtr[s.first % chSize], regPtr[chSize + s.first % chSize]});
    }

    return reg;
}

std::vector<std::pair<float, float>> filterWH(const ov::Tensor& whTensor,
                                              const std::vector<std::pair<size_t, float>>& scores,
                                              size_t chSize) {
    const float* whPtr = whTensor.data<float>();
    std::vector<std::pair<float, float>> wh;

    for (auto s : scores) {
        wh.push_back({whPtr[s.first % chSize], whPtr[chSize + s.first % chSize]});
    }

    return wh;
}

std::vector<ModelCenterNet::BBox> calcBoxes(const std::vector<std::pair<size_t, float>>& scores,
                                            const std::vector<std::pair<float, float>>& reg,
                                            const std::vector<std::pair<float, float>>& wh,
                                            const ov::Shape& shape) {
    std::vector<ModelCenterNet::BBox> boxes(scores.size());

    for (size_t i = 0; i < boxes.size(); ++i) {
        size_t chIdx = scores[i].first % (shape[2] * shape[3]);
        auto xCenter = chIdx % shape[3];
        auto yCenter = chIdx / shape[3];

        boxes[i].left = xCenter + reg[i].first - wh[i].first / 2.0f;
        boxes[i].top = yCenter + reg[i].second - wh[i].second / 2.0f;
        boxes[i].right = xCenter + reg[i].first + wh[i].first / 2.0f;
        boxes[i].bottom = yCenter + reg[i].second + wh[i].second / 2.0f;
    }

    return boxes;
}

void transform(std::vector<ModelCenterNet::BBox>& boxes,
               const ov::Shape& shape,
               int scale,
               float centerX,
               float centerY) {
    cv::Mat1f trans = getAffineTransform(centerX, centerY, scale, 0, shape[2], shape[3], true);

    for (auto& b : boxes) {
        ModelCenterNet::BBox newbb;

        newbb.left = trans.at<float>(0, 0) * b.left + trans.at<float>(0, 1) * b.top + trans.at<float>(0, 2);
        newbb.top = trans.at<float>(1, 0) * b.left + trans.at<float>(1, 1) * b.top + trans.at<float>(1, 2);
        newbb.right = trans.at<float>(0, 0) * b.right + trans.at<float>(0, 1) * b.bottom + trans.at<float>(0, 2);
        newbb.bottom = trans.at<float>(1, 0) * b.right + trans.at<float>(1, 1) * b.bottom + trans.at<float>(1, 2);

        b = newbb;
    }
}

std::unique_ptr<ResultBase> ModelCenterNet::postprocess(InferenceResult& infResult) {
    // --------------------------- Filter data and get valid indices ---------------------------------
    const auto& heatmapTensor = infResult.outputsData[outputsNames[0]];
    const auto& heatmapTensorShape = heatmapTensor.get_shape();
    const auto chSize = heatmapTensorShape[2] * heatmapTensorShape[3];
    const auto scores = filterScores(heatmapTensor, confidenceThreshold);

    const auto& regressionTensor = infResult.outputsData[outputsNames[1]];
    const auto reg = filterReg(regressionTensor, scores, chSize);

    const auto& whTensor = infResult.outputsData[outputsNames[2]];
    const auto wh = filterWH(whTensor, scores, chSize);

    // --------------------------- Calculate bounding boxes & apply inverse affine transform ----------
    auto boxes = calcBoxes(scores, reg, wh, heatmapTensorShape);

    const auto imgWidth = infResult.internalModelData->asRef<InternalImageModelData>().inputImgWidth;
    const auto imgHeight = infResult.internalModelData->asRef<InternalImageModelData>().inputImgHeight;
    const auto scale = std::max(imgWidth, imgHeight);
    const float centerX = imgWidth / 2.0f;
    const float centerY = imgHeight / 2.0f;

    transform(boxes, heatmapTensorShape, scale, centerX, centerY);

    // --------------------------- Create detection result objects ------------------------------------
    DetectionResult* result = new DetectionResult(infResult.frameId, infResult.metaData);

    result->objects.reserve(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        DetectedObject desc;
        desc.confidence = scores[i].second;
        desc.labelID = scores[i].first / chSize;
        desc.label = getLabelName(desc.labelID);
        desc.x = clamp(boxes[i].left, 0.f, static_cast<float>(imgWidth));
        desc.y = clamp(boxes[i].top, 0.f, static_cast<float>(imgHeight));
        desc.width = clamp(boxes[i].getWidth(), 0.f, static_cast<float>(imgWidth));
        desc.height = clamp(boxes[i].getHeight(), 0.f, static_cast<float>(imgHeight));

        result->objects.push_back(desc);
    }

    return std::unique_ptr<ResultBase>(result);
}
