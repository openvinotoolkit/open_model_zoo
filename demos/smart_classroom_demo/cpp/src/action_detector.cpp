// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>

#include "openvino/openvino.hpp"

#include "action_detector.hpp"

#define SSD_LOCATION_RECORD_SIZE 4
#define SSD_PRIORBOX_RECORD_SIZE 4
#define NUM_DETECTION_CLASSES 2
#define POSITIVE_DETECTION_IDX 1

template <typename T>
bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2) {
    return pair1.first > pair2.first;
}

ActionDetection::ActionDetection(const ActionDetectorConfig& config) :
    BaseCnnDetection(config.is_async), m_config(config) {
    m_detectorName = "action detector";

    ov::Layout desiredLayout = {"NHWC"};

    slog::info << "Reading model: " << m_config.m_path_to_model << slog::endl;
    std::shared_ptr<ov::Model> model = m_config.m_core.read_model(m_config.m_path_to_model);
    logBasicModelInfo(model);

    ov::OutputVector inputInfo = model->inputs();
    if (inputInfo.size() != 1) {
        throw std::runtime_error("Action Detection network should have only one input");
    }

    m_input_name = model->input().get_any_name();

    m_modelLayout = ov::layout::get_layout(model->input());
    if (m_modelLayout.empty())
        m_modelLayout = {"NCHW"};

    m_network_input_size.height = model->input().get_shape()[ov::layout::height_idx(m_modelLayout)];
    m_network_input_size.width = model->input().get_shape()[ov::layout::width_idx(m_modelLayout)];

    m_new_model = false;

    ov::OutputVector outputs = model->outputs();
    auto cmp = [&](const ov::Output<ov::Node>& output) { return output.get_any_name() == m_config.new_loc_blob_name; };
    auto it = std::find_if(outputs.begin(), outputs.end(), cmp);
    if (it != outputs.end())
        m_new_model = true;

    ov::preprocess::PrePostProcessor ppp(model);

    ppp.input().tensor()
        .set_element_type(ov::element::u8)
        .set_layout(desiredLayout);
    ppp.input().preprocess()
        .convert_layout(m_modelLayout)
        .convert_element_type(ov::element::f32);
    ppp.input().model().set_layout(m_modelLayout);

    model = ppp.build();

    slog::info << "PrePostProcessor configuration:" << slog::endl;
    slog::info << ppp << slog::endl;

    ov::set_batch(model, m_config.m_max_batch_size);

    m_model = m_config.m_core.compile_model(model, m_config.m_deviceName);
    logCompiledModelInfo(m_model, m_config.m_path_to_model, m_config.m_deviceName, m_config.m_model_type);

    const auto& head_anchors = m_new_model ? m_config.new_anchors : m_config.old_anchors;
    const int num_heads = head_anchors.size();

    m_head_ranges.resize(num_heads + 1);
    m_glob_anchor_map.resize(num_heads);
    m_head_step_sizes.resize(num_heads);

    m_head_ranges[0] = 0;
    int head_shift = 0;

    for (int head_id = 0; head_id < num_heads; ++head_id) {
        m_glob_anchor_map[head_id].resize(head_anchors[head_id]);
        int anchor_height, anchor_width;
        for (int anchor_id = 0; anchor_id < head_anchors[head_id]; ++anchor_id) {
            const auto glob_anchor_name = m_new_model ?
                m_config.new_action_conf_blob_name_prefix + std::to_string(head_id + 1) +
                    m_config.new_action_conf_blob_name_suffix + std::to_string(anchor_id + 1) :
                m_config.old_action_conf_blob_name_prefix + std::to_string(anchor_id + 1);
            m_glob_anchor_names.push_back(glob_anchor_name);

            const auto anchor_dims = m_model.output(glob_anchor_name).get_shape();

            anchor_height = m_new_model ? anchor_dims[ov::layout::height_idx(m_modelLayout)] : anchor_dims[1];
            anchor_width = m_new_model ? anchor_dims[ov::layout::width_idx(m_modelLayout)] : anchor_dims[2];
            std::size_t num_action_classes = m_new_model ? anchor_dims[ov::layout::channels_idx(m_modelLayout)] : anchor_dims[3];
            if (num_action_classes != m_config.num_action_classes) {
                throw std::logic_error("The number of specified actions and the number of actions predicted by "
                    "the Person/Action Detection Retail model must match");
            }

            const int anchor_size = anchor_height * anchor_width;
            head_shift += anchor_size;

            m_head_step_sizes[head_id] = m_new_model ? anchor_size : 1;
            m_glob_anchor_map[head_id][anchor_id] = m_num_glob_anchors++;
        }

        m_head_ranges[head_id + 1] = head_shift;
        m_head_blob_sizes.emplace_back(anchor_width, anchor_height);
    }

    m_num_candidates = head_shift;

    m_binary_task = m_config.num_action_classes == 2;
}

void ActionDetection::submitRequest() {
    if (!m_enqueued_frames)
        return;
    m_enqueued_frames = 0;

    BaseCnnDetection::submitRequest();
}

void ActionDetection::enqueue(const cv::Mat& frame) {
    if (m_request == nullptr) {
        m_request = std::make_shared<ov::InferRequest>(m_model.create_infer_request());
    }

    m_width = static_cast<float>(frame.cols);
    m_height = static_cast<float>(frame.rows);

    ov::Tensor input_tensor = m_request->get_tensor(m_input_name);

    resize2tensor(frame, input_tensor);

    m_enqueued_frames = 1;
}

DetectedActions ActionDetection::fetchResults() {
    const auto loc_blob_name = m_new_model ? m_config.new_loc_blob_name : m_config.old_loc_blob_name;
    const auto det_conf_blob_name = m_new_model ? m_config.new_det_conf_blob_name : m_config.old_det_conf_blob_name;

    ov::Shape loc_out_shape = m_model.output(loc_blob_name).get_shape();
    const cv::Mat loc_out(loc_out_shape[0],
                          loc_out_shape[1],
                          CV_32F,
                          m_request->get_tensor(loc_blob_name).data<float>());

    ov::Shape conf_out_shape = m_model.output(det_conf_blob_name).get_shape();
    const cv::Mat main_conf_out(conf_out_shape[0],
                                conf_out_shape[1],
                                CV_32F,
                                m_request->get_tensor(det_conf_blob_name).data<float>());

    std::vector<cv::Mat> add_conf_out;
    for (int glob_anchor_id = 0; glob_anchor_id < m_num_glob_anchors; ++glob_anchor_id) {
        const auto& blob_name = m_glob_anchor_names[glob_anchor_id];
        ov::Shape shape = m_request->get_tensor(blob_name).get_shape();
        add_conf_out.emplace_back(shape[ov::layout::height_idx(m_modelLayout)],
                                  shape[ov::layout::width_idx(m_modelLayout)],
                                  CV_32F,
                                  m_request->get_tensor(blob_name).data());
    }

    // Parse detections
    DetectedActions result;
    if (m_new_model) {
        const cv::Mat priorbox_out;
        result = GetDetections(loc_out, main_conf_out, priorbox_out, add_conf_out,
                             cv::Size(static_cast<int>(m_width), static_cast<int>(m_height)));
    } else {
        ov::Shape old_priorbox_shape = m_model.output(m_config.old_priorbox_blob_name).get_shape();
        const cv::Mat priorbox_out((int)old_priorbox_shape[0], (int)old_priorbox_shape[1],
            CV_32F,
            m_request->get_tensor(m_config.old_priorbox_blob_name).data());

        result = GetDetections(loc_out, main_conf_out, priorbox_out, add_conf_out,
            cv::Size(static_cast<int>(m_width), static_cast<int>(m_height)));
    }

    return result;
}

inline ActionDetection::NormalizedBBox
ActionDetection::ParseBBoxRecord(const float* data, bool inverse) const {
    NormalizedBBox bbox;
    bbox.xmin = data[inverse ? 1 : 0];
    bbox.ymin = data[inverse ? 0 : 1];
    bbox.xmax = data[inverse ? 3 : 2];
    bbox.ymax = data[inverse ? 2 : 3];
    return bbox;
}

inline ActionDetection::NormalizedBBox
ActionDetection::GeneratePriorBox(int pos, int step, const cv::Size2f& anchor,
                                  const cv::Size& blob_size) const {
    const int row = pos / blob_size.width;
    const int col = pos % blob_size.width;

    const float center_x = (col + 0.5f) * static_cast<float>(step);
    const float center_y = (row + 0.5f) * static_cast<float>(step);

    NormalizedBBox bbox;
    bbox.xmin = (center_x - 0.5f * anchor.width) / static_cast<float>(m_network_input_size.width);
    bbox.ymin = (center_y - 0.5f * anchor.height) / static_cast<float>(m_network_input_size.height);
    bbox.xmax = (center_x + 0.5f * anchor.width) / static_cast<float>(m_network_input_size.width);
    bbox.ymax = (center_y + 0.5f * anchor.height) / static_cast<float>(m_network_input_size.height);

    return bbox;
}

cv::Rect ActionDetection::ConvertToRect(
        const NormalizedBBox& prior_bbox, const NormalizedBBox& variances,
        const NormalizedBBox& encoded_bbox, const cv::Size& frame_size) const {
    // Convert prior bbox to CV_Rect
    const float prior_width = prior_bbox.xmax - prior_bbox.xmin;
    const float prior_height = prior_bbox.ymax - prior_bbox.ymin;
    const float prior_center_x = 0.5f * (prior_bbox.xmin + prior_bbox.xmax);
    const float prior_center_y = 0.5f * (prior_bbox.ymin + prior_bbox.ymax);

    // Decode bbox coordinates from the SSD format
    const float decoded_bbox_center_x =
            variances.xmin * encoded_bbox.xmin * prior_width + prior_center_x;
    const float decoded_bbox_center_y =
            variances.ymin * encoded_bbox.ymin * prior_height + prior_center_y;
    const float decoded_bbox_width =
            static_cast<float>(exp(static_cast<float>(variances.xmax * encoded_bbox.xmax))) * prior_width;
    const float decoded_bbox_height =
            static_cast<float>(exp(static_cast<float>(variances.ymax * encoded_bbox.ymax))) * prior_height;

    // Create decoded bbox
    const float decoded_bbox_xmin = decoded_bbox_center_x - 0.5f * decoded_bbox_width;
    const float decoded_bbox_ymin = decoded_bbox_center_y - 0.5f * decoded_bbox_height;
    const float decoded_bbox_xmax = decoded_bbox_center_x + 0.5f * decoded_bbox_width;
    const float decoded_bbox_ymax = decoded_bbox_center_y + 0.5f * decoded_bbox_height;

    // Convert decoded bbox to CV_Rect
    return cv::Rect(static_cast<int>(decoded_bbox_xmin * frame_size.width),
                    static_cast<int>(decoded_bbox_ymin * frame_size.height),
                    static_cast<int>((decoded_bbox_xmax - decoded_bbox_xmin) * frame_size.width),
                    static_cast<int>((decoded_bbox_ymax - decoded_bbox_ymin) * frame_size.height));
}

DetectedActions ActionDetection::GetDetections(const cv::Mat& loc, const cv::Mat& main_conf,
        const cv::Mat& priorboxes, const std::vector<cv::Mat>& add_conf,
        const cv::Size& frame_size) const {
    // Prepare input data buffers
    const float* loc_data = reinterpret_cast<float*>(loc.data);
    const float* det_conf_data = reinterpret_cast<float*>(main_conf.data);
    const float* prior_data = m_new_model ? NULL : reinterpret_cast<float*>(priorboxes.data);

    const int total_num_anchors = add_conf.size();
    std::vector<float*> action_conf_data(total_num_anchors);
    for (int i = 0; i < total_num_anchors; ++i) {
        action_conf_data[i] = reinterpret_cast<float*>(add_conf[i].data);
    }

    // Variable to store all detection candidates
    DetectedActions valid_detections;

    // Iterate over all candidate bboxes
    for (int p = 0; p < m_num_candidates; ++p) {
        // Parse detection confidence from the SSD Detection output
        const float detection_conf =
                det_conf_data[p * NUM_DETECTION_CLASSES + POSITIVE_DETECTION_IDX];

        // Skip low-confidence detections
        if (detection_conf < m_config.detection_confidence_threshold) {
            continue;
        }

        // Estimate the action head ID
        int head_id = 0;
        while (p >= m_head_ranges[head_id + 1]) {
            ++head_id;
        }
        const int head_p = p - m_head_ranges[head_id];

        // Estimate the action anchor ID
        const int head_num_anchors =
            m_new_model ? m_config.new_anchors[head_id] : m_config.old_anchors[head_id];
        const int anchor_id = head_p % head_num_anchors;

        // Estimate the action label
        const int glob_anchor_id = m_glob_anchor_map[head_id][anchor_id];
        const float* anchor_conf_data = action_conf_data[glob_anchor_id];
        const int action_conf_idx_shift = m_new_model ?
            head_p / head_num_anchors :
            head_p / head_num_anchors * m_config.num_action_classes;
        const int action_conf_step = m_head_step_sizes[head_id];
        const float scale = m_new_model ? m_config.new_action_scale : m_config.old_action_scale;
        int action_label = -1;
        float action_max_exp_value = 0.f;
        float action_sum_exp_values = 0.f;
        for (size_t c = 0; c < m_config.num_action_classes; ++c) {
            float action_exp_value =
                std::exp(scale * anchor_conf_data[action_conf_idx_shift + c * action_conf_step]);
            action_sum_exp_values += action_exp_value;
            if (action_exp_value > action_max_exp_value && ((c > 0 && m_binary_task) || !m_binary_task)) {
                action_max_exp_value = action_exp_value;
                action_label = c;
            }
        }

        if (std::fabs(action_sum_exp_values) < std::numeric_limits<float>::epsilon()) {
            throw std::logic_error("action_sum_exp_values can't be equal to 0");
        }
        // Estimate the action confidence
        float action_conf = action_max_exp_value / action_sum_exp_values;

        // Skip low-confidence actions
        if (action_label < 0 || action_conf < m_config.action_confidence_threshold) {
            action_label = m_config.default_action_id;
            action_conf = 0.f;
        }

        // Parse bbox from the SSD Detection output
        const auto priorbox = m_new_model ?
            GeneratePriorBox(head_p / head_num_anchors,
                m_config.new_det_heads[head_id].step,
                m_config.new_det_heads[head_id].anchors[anchor_id],
                m_head_blob_sizes[head_id]) :
            ParseBBoxRecord(prior_data + p * SSD_PRIORBOX_RECORD_SIZE, false);
        const auto variance =
                ParseBBoxRecord(m_new_model ?
                    m_config.variances :
                    prior_data + (m_num_candidates + p) * SSD_PRIORBOX_RECORD_SIZE, false);
        const auto encoded_bbox =
                ParseBBoxRecord(loc_data + p * SSD_LOCATION_RECORD_SIZE, m_new_model);

        const auto det_rect = ConvertToRect(priorbox, variance, encoded_bbox, frame_size);

        // Store detected action
        valid_detections.emplace_back(det_rect, action_label, detection_conf, action_conf);
    }

    // Merge most overlapped detections
    std::vector<int> out_det_indices;
    SoftNonMaxSuppression(valid_detections, m_config.nms_sigma, m_config.keep_top_k,
                          m_config.detection_confidence_threshold, &out_det_indices);

    DetectedActions detections;
    for (size_t i = 0; i < out_det_indices.size(); ++i) {
        detections.emplace_back(valid_detections[out_det_indices[i]]);
    }
    return detections;
}

void ActionDetection::SoftNonMaxSuppression(const DetectedActions& detections,
        const float sigma, size_t top_k, const float min_det_conf,
        std::vector<int>* out_indices) const {
    // Store input bbox scores
    std::vector<float> scores(detections.size());
    for (size_t i = 0; i < detections.size(); ++i) {
        scores[i] = detections[i].detection_conf;
    }

    top_k = std::min(top_k, scores.size());

    // Select top-k score indices
    std::vector<size_t> score_idx(scores.size());
    std::iota(score_idx.begin(), score_idx.end(), 0);
    std::nth_element(score_idx.begin(), score_idx.begin() + top_k, score_idx.end(),
        [&scores](size_t i1, size_t i2) {return scores[i1] > scores[i2];});

    // Extract top-k score values
    std::vector<float> top_scores(top_k);
    for (size_t i = 0; i < top_scores.size(); ++i) {
        top_scores[i] = scores[score_idx[i]];
    }

    // Carry out Soft Non-Maximum Suppression algorithm
    out_indices->clear();
    for (size_t step = 0; step < top_scores.size(); ++step) {
        auto best_score_itr = std::max_element(top_scores.begin(), top_scores.end());
        if (*best_score_itr < min_det_conf) {
            break;
        }

        // Add current bbox to output list
        const size_t local_anchor_idx = std::distance(top_scores.begin(), best_score_itr);
        const int anchor_idx = score_idx[local_anchor_idx];
        out_indices->emplace_back(anchor_idx);
        *best_score_itr = 0.f;

        // Update top_scores of the rest bboxes
        for (size_t local_reference_idx = 0; local_reference_idx < top_scores.size(); ++local_reference_idx) {
            // Skip updating step for the low-confidence bbox
            if (top_scores[local_reference_idx] < min_det_conf) {
                continue;
            }

            // Calculate the Intersection over Union metric between two bboxes
            const size_t reference_idx = score_idx[local_reference_idx];
            const auto& rect1 = detections[anchor_idx].rect;
            const auto& rect2 = detections[reference_idx].rect;
            const auto intersection = rect1 & rect2;
            float overlap = 0.f;
            if (intersection.width > 0 && intersection.height > 0) {
                const int intersection_area = intersection.area();
                overlap = static_cast<float>(intersection_area) / static_cast<float>(rect1.area() + rect2.area() - intersection_area);
            }

            // Scale bbox score using the exponential rule
            top_scores[local_reference_idx] *= std::exp(-overlap * overlap / sigma);
        }
    }
}
