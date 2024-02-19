// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <string>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "openvino/openvino.hpp"

#include "detector.hpp"

#define SSD_EMPTY_DETECTIONS_INDICATOR -1.0

using namespace detection;

namespace {
cv::Rect TruncateToValidRect(const cv::Rect& rect, const cv::Size& size) {
    auto tl = rect.tl(), br = rect.br();
    tl.x = std::max(0, std::min(size.width - 1, tl.x));
    tl.y = std::max(0, std::min(size.height - 1, tl.y));
    br.x = std::max(0, std::min(size.width, br.x));
    br.y = std::max(0, std::min(size.height, br.y));
    int w = std::max(0, br.x - tl.x);
    int h = std::max(0, br.y - tl.y);
    return cv::Rect(tl.x, tl.y, w, h);
}

cv::Rect IncreaseRect(const cv::Rect& r, float coeff_x, float coeff_y) {
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

FaceDetection::FaceDetection(const DetectorConfig& config) :
        BaseCnnDetection(config.is_async), m_config(config) {
    m_detectorName = "Face Detection";

    slog::info << "Reading model: " << m_config.m_path_to_model << slog::endl;
    std::shared_ptr<ov::Model> model = m_config.m_core.read_model(m_config.m_path_to_model);
    logBasicModelInfo(model);

    ov::Layout desiredLayout = {"NHWC"};

    ov::OutputVector inputs_info = model->inputs();
    if (inputs_info.size() != 1) {
        throw std::runtime_error("Face Detection network should have only one input");
    }

    ov::OutputVector outputs_info = model->outputs();
    if (outputs_info.size() != 1) {
        throw std::runtime_error("Face Detection network should have only one output");
    }

    ov::Output<ov::Node> input = model->input();
    m_input_name = input.get_any_name();

    ov::Layout modelLayout = ov::layout::get_layout(input);
    if (modelLayout.empty())
        modelLayout = {"NCHW"};

    ov::Shape shape = input.get_shape();
    shape[ov::layout::height_idx(modelLayout)] = m_config.input_h;
    shape[ov::layout::width_idx(modelLayout)] = m_config.input_w;

    ov::Output<ov::Node> output = model->output();
    m_output_name = output.get_any_name();

    ov::Shape outputDims = output.get_shape();
    m_max_detections_count = outputDims[2];
    m_object_size = outputDims[3];
    if (m_object_size != 7) {
        throw std::runtime_error("Face Detection network output layer should have 7 as a last dimension");
    }
    if (outputDims.size() != 4) {
        throw std::runtime_error("Face Detection network output should have 4 dimensions, but had " +
            std::to_string(outputDims.size()));
    }

    std::map<std::string, ov::PartialShape> input_shapes;
    input_shapes[input.get_any_name()] = shape;
    model->reshape(input_shapes);

    ov::preprocess::PrePostProcessor ppp(model);

    ppp.input().tensor()
        .set_element_type(ov::element::u8)
        .set_layout(desiredLayout);
    ppp.input().preprocess()
        .convert_layout(modelLayout)
        .convert_element_type(ov::element::f32);
    ppp.input().model().set_layout(modelLayout);

    model = ppp.build();

    slog::info << "PrePostProcessor configuration:" << slog::endl;
    slog::info << ppp << slog::endl;

    m_model = m_config.m_core.compile_model(model, m_config.m_deviceName);
    logCompiledModelInfo(m_model, m_config.m_path_to_model, m_config.m_deviceName, m_detectorName);
}

void FaceDetection::submitRequest() {
    if (!m_enqueued_frames)
        return;
    m_enqueued_frames = 0;

    BaseCnnDetection::submitRequest();
}

void FaceDetection::enqueue(const cv::Mat& frame) {
    if (m_request == nullptr) {
        m_request = std::make_shared<ov::InferRequest>(m_model.create_infer_request());
    }

    m_width = static_cast<float>(frame.cols);
    m_height = static_cast<float>(frame.rows);

    ov::Tensor inputTensor = m_request->get_tensor(m_input_name);

    resize2tensor(frame, inputTensor);

    m_enqueued_frames = 1;
}

DetectedObjects FaceDetection::fetchResults() {
    DetectedObjects results;
    const float* data = m_request->get_tensor(m_output_name).data<float>();

    for (int det_id = 0; det_id < m_max_detections_count; ++det_id) {
        const int start_pos = det_id * m_object_size;

        const float batchID = data[start_pos];
        if (batchID == SSD_EMPTY_DETECTIONS_INDICATOR) {
            break;
        }

        const float score = std::min(std::max(0.0f, data[start_pos + 2]), 1.0f);
        const float x0 = std::min(std::max(0.0f, data[start_pos + 3]), 1.0f) * m_width;
        const float y0 = std::min(std::max(0.0f, data[start_pos + 4]), 1.0f) * m_height;
        const float x1 = std::min(std::max(0.0f, data[start_pos + 5]), 1.0f) * m_width;
        const float y1 = std::min(std::max(0.0f, data[start_pos + 6]), 1.0f) * m_height;

        DetectedObject object;
        object.confidence = score;
        object.rect = cv::Rect(cv::Point(static_cast<int>(round(static_cast<double>(x0))),
                                         static_cast<int>(round(static_cast<double>(y0)))),
                               cv::Point(static_cast<int>(round(static_cast<double>(x1))),
                                         static_cast<int>(round(static_cast<double>(y1)))));


        object.rect = TruncateToValidRect(IncreaseRect(object.rect,
                                                       m_config.increase_scale_x,
                                                       m_config.increase_scale_y),
                                          cv::Size(static_cast<int>(m_width), static_cast<int>(m_height)));

        if (object.confidence > m_config.confidence_threshold && object.rect.area() > 0) {
            results.emplace_back(object);
        }
    }

    return results;
}
