// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detector.hpp"

#include <algorithm>
#include <string>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <inference_engine.hpp>

#include <ngraph/ngraph.hpp>

#include "openvino/openvino.hpp"

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
    BaseCnnDetection::submitRequest();
}

void FaceDetection::enqueue(const cv::Mat &frame) {
    if (request == nullptr) {
        request = std::make_shared<ov::runtime::InferRequest>(model_.create_infer_request());
    }

    width_ = static_cast<float>(frame.cols);
    height_ = static_cast<float>(frame.rows);

    ov::runtime::Tensor inputTensor = request->get_tensor(input_name_);

    matToTensor(frame, inputTensor);

    enqueued_frames_ = 1;
}

FaceDetection::FaceDetection(const DetectorConfig& config) :
        BaseCnnDetection(config.is_async), config_(config) {
    topoName = "Face Detection";
    auto cnnNetwork = config.ie.read_model(config.path_to_model);

    ov::OutputVector inputs_info = cnnNetwork->inputs();
    if (inputs_info.size() != 1) {
        throw std::runtime_error("Face Detection network should have only one input");
    }

    ov::preprocess::PrePostProcessor proc(cnnNetwork);
    ov::preprocess::InputInfo& input_info = proc.input();
    input_info.tensor().set_element_type(ov::element::u8).set_layout({ "NCHW" });

    ov::Output<ov::Node> _input = cnnNetwork->input();
    ov::Shape input_dims = _input.get_shape();
    input_dims[2] = config_.input_h;
    input_dims[3] = config_.input_w;
    std::map<std::string, ov::PartialShape> input_shapes;
    input_shapes[_input.get_any_name()] = input_dims;
    cnnNetwork->reshape(input_shapes);

    ov::OutputVector outputs_info = cnnNetwork->outputs();
    if (outputs_info.size() != 1) {
        throw std::runtime_error("Face Detection network should have only one output");
    }
    ov::Output<ov::Node> _output = cnnNetwork->output();
    output_name_ = _output.get_any_name();

    ov::Shape outputDims = _output.get_shape();
    max_detections_count_ = outputDims[2];
    object_size_ = outputDims[3];
    if (object_size_ != 7) {
        throw std::runtime_error("Face Detection network output layer should have 7 as a last dimension");
    }
    if (outputDims.size() != 4) {
        throw std::runtime_error("Face Detection network output should have 4 dimensions, but had " +
            std::to_string(outputDims.size()));
    }

    ov::preprocess::OutputInfo& output_info = proc.output();
    output_info.tensor().set_element_type(ov::element::f32);

    input_name_ = cnnNetwork->input().get_any_name();
    cnnNetwork = proc.build();
    model_ = config_.ie.compile_model(cnnNetwork, config_.deviceName);

    logExecNetworkInfo(model_, config_.path_to_model, config_.deviceName, topoName);
}

DetectedObjects FaceDetection::fetchResults() {
    DetectedObjects results;
    const float* data = request->get_tensor(output_name_).data<float>();

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
        object.rect = cv::Rect(cv::Point(static_cast<int>(round(static_cast<double>(x0))),
                                         static_cast<int>(round(static_cast<double>(y0)))),
                               cv::Point(static_cast<int>(round(static_cast<double>(x1))),
                                         static_cast<int>(round(static_cast<double>(y1)))));


        object.rect = TruncateToValidRect(IncreaseRect(object.rect,
                                                       config_.increase_scale_x,
                                                       config_.increase_scale_y),
                                          cv::Size(static_cast<int>(width_), static_cast<int>(height_)));

        if (object.confidence > config_.confidence_threshold && object.rect.area() > 0) {
            results.emplace_back(object);
        }
    }

    return results;
}
