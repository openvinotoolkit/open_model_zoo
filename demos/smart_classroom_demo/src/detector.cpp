// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detector.hpp"

#include <algorithm>
#include <string>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <inference_engine.hpp>

using namespace InferenceEngine;

#define SSD_EMPTY_DETECTIONS_INDICATOR -1.0

using namespace detection;

namespace {
cv::Rect TruncateToValidRect(const cv::Rect& rect,
                             const cv::Size& size) {
    auto tl = rect.tl(), br = rect.br();
    tl.x = std::max(0, std::min(size.width - 1, tl.x));
    tl.y = std::max(0, std::min(size.height - 1, tl.y));
    br.x = std::max(0, std::min(size.width, br.x));
    br.y = std::max(0, std::min(size.height, br.y));
    int w = std::max(0, br.x - tl.x);
    int h = std::max(0, br.y - tl.y);
    return cv::Rect(tl.x, tl.y, w, h);
}

cv::Rect IncreaseRect(const cv::Rect& r, float coeff_x,
                      float coeff_y)  {
    cv::Point2f tl = r.tl();
    cv::Point2f br = r.br();
    cv::Point2f c = (tl * 0.5f) + (br * 0.5f);
    cv::Point2f diff = c - tl;
    cv::Point2f new_diff{diff.x * coeff_x, diff.y * coeff_y};
    cv::Point2f new_tl = c - new_diff;
    cv::Point2f new_br = c + new_diff;

    cv::Point new_tl_int {static_cast<int>(std::floor(new_tl.x)), static_cast<int>(std::floor(new_tl.y))};
    cv::Point new_br_int {static_cast<int>(std::ceil(new_br.x)), static_cast<int>(std::ceil(new_br.y))};

    return cv::Rect(new_tl_int, new_br_int);
}
}  // namespace

void FaceDetection::submitRequest() {
    if (!enqueued_frames_) return;
    enqueued_frames_ = 0;
    results_fetched_ = false;
    results.clear();
    BaseCnnDetection::submitRequest();
}

void FaceDetection::enqueue(const cv::Mat &frame) {
    if (!enabled()) return;

    if (!request) {
        request = net_.CreateInferRequestPtr();
    }

    width_ = frame.cols;
    height_ = frame.rows;

    Blob::Ptr inputBlob = request->GetBlob(input_name_);

    matU8ToBlob<uint8_t>(frame, inputBlob);

    enqueued_frames_ = 1;
}

FaceDetection::FaceDetection(const DetectorConfig& config) :
    BaseCnnDetection(config.enabled, config.is_async), config_(config) {
    if (config.enabled) {
        topoName = "face detector";
        CNNNetReader net_reader;
        net_reader.ReadNetwork(config.path_to_model);
        net_reader.ReadWeights(config.path_to_weights);
        if (!net_reader.isParseSuccess()) {
            THROW_IE_EXCEPTION << "Cannot load model";
        }

        InputsDataMap inputInfo(net_reader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            THROW_IE_EXCEPTION << "Face Detection network should have only one input";
        }
        InputInfo::Ptr inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setPrecision(Precision::U8);
        inputInfoFirst->getInputData()->setLayout(Layout::NCHW);

        SizeVector input_dims = inputInfoFirst->getInputData()->getTensorDesc().getDims();
        input_dims[2] = config_.input_h;
        input_dims[3] = config_.input_w;
        std::map<std::string, SizeVector> input_shapes;
        input_shapes[inputInfo.begin()->first] = input_dims;
        net_reader.getNetwork().reshape(input_shapes);

        OutputsDataMap outputInfo(net_reader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 1) {
            THROW_IE_EXCEPTION << "Face Detection network should have only one output";
        }
        DataPtr& _output = outputInfo.begin()->second;
        output_name_ = outputInfo.begin()->first;

        const CNNLayerPtr outputLayer = net_reader.getNetwork().getLayerByName(output_name_.c_str());
        if (outputLayer->type != "DetectionOutput") {
            THROW_IE_EXCEPTION << "Face Detection network output layer(" + outputLayer->name +
                                  ") should be DetectionOutput, but was " +  outputLayer->type;
        }

        if (outputLayer->params.find("num_classes") == outputLayer->params.end()) {
            THROW_IE_EXCEPTION << "Face Detection network output layer (" +
                                  output_name_ + ") should have num_classes integer attribute";
        }

        const SizeVector outputDims = _output->getTensorDesc().getDims();
        max_detections_count_ = outputDims[2];
        object_size_ = outputDims[3];
        if (object_size_ != 7) {
            THROW_IE_EXCEPTION << "Face Detection network output layer should have 7 as a last dimension";
        }
        if (outputDims.size() != 4) {
            THROW_IE_EXCEPTION << "Face Detection network output dimensions not compatible shoulld be 4, but was " +
                                  std::to_string(outputDims.size());
        }
        _output->setPrecision(Precision::FP32);
        _output->setLayout(TensorDesc::getLayoutByDims(_output->getDims()));

        input_name_ = inputInfo.begin()->first;
        net_ = config_.plugin.LoadNetwork(net_reader.getNetwork(), {});
    }
}

void FaceDetection::fetchResults() {
    if (!enabled()) return;
    results.clear();
    if (results_fetched_) return;
    results_fetched_ = true;
    const float *data = request->GetBlob(output_name_)->buffer().as<float *>();

    for (int det_id = 0; det_id < max_detections_count_; ++det_id) {
        const int start_pos = det_id * object_size_;

        const float batchID = data[start_pos];
        if (batchID == SSD_EMPTY_DETECTIONS_INDICATOR) {
            break;
        }

        const float score = std::min(std::max(0.0f, data[start_pos + 2]), 1.0f);
        const float x0 =
                std::min(std::max(0.0f, data[start_pos + 3]), 1.0f) * width_;
        const float y0 =
                std::min(std::max(0.0f, data[start_pos + 4]), 1.0f) * height_;
        const float x1 =
                std::min(std::max(0.0f, data[start_pos + 5]), 1.0f) * width_;
        const float y1 =
                std::min(std::max(0.0f, data[start_pos + 6]), 1.0f) * height_;

        DetectedObject object;
        object.confidence = score;
        object.rect = cv::Rect(cv::Point(round(x0), round(y0)),
                               cv::Point(round(x1), round(y1)));

        object.rect = TruncateToValidRect(IncreaseRect(object.rect,
                                                       config_.increase_scale_x,
                                                       config_.increase_scale_y),
                                          cv::Size(width_, height_));

        if (object.confidence > config_.confidence_threshold && object.rect.area() > 0) {
            results.emplace_back(object);
        }
    }
}
