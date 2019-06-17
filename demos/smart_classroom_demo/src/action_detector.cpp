// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "action_detector.hpp"
#include <algorithm>
#include <utility>
#include <vector>
#include <limits>
#include <opencv2/imgproc/imgproc.hpp>

using namespace InferenceEngine;

#define SSD_LOCATION_RECORD_SIZE 4
#define SSD_PRIORBOX_RECORD_SIZE 4
#define NUM_DETECTION_CLASSES 2
#define POSITIVE_DETECTION_IDX 1
#define INVALID_TOP_K_IDX -1

template <typename T>
bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2) {
    return pair1.first > pair2.first;
}

void ActionDetection::submitRequest() {
    if (!enqueued_frames_) return;
    enqueued_frames_ = 0;
    results_fetched_ = false;
    results.clear();
    BaseCnnDetection::submitRequest();
}

void ActionDetection::enqueue(const cv::Mat &frame) {
    if (!enabled()) return;

    if (!request) {
        request = net_.CreateInferRequestPtr();
    }

    width_ = static_cast<float>(frame.cols);
    height_ = static_cast<float>(frame.rows);

    Blob::Ptr inputBlob = request->GetBlob(input_name_);

    matU8ToBlob<uint8_t>(frame, inputBlob);

    enqueued_frames_ = 1;
}

ActionDetection::ActionDetection(const ActionDetectorConfig& config)
    : BaseCnnDetection(config.enabled, config.is_async), config_(config) {
    if (config.enabled) {
        topoName = "action detector";
        CNNNetReader net_reader;
        net_reader.ReadNetwork(config.path_to_model);
        net_reader.ReadWeights(config.path_to_weights);
        if (!net_reader.isParseSuccess()) {
            THROW_IE_EXCEPTION << "Cannot load model";
        }

        net_reader.getNetwork().setBatchSize(config.max_batch_size);

        InputsDataMap inputInfo(net_reader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            THROW_IE_EXCEPTION << "Action Detection network should have only one input";
        }
        InputInfo::Ptr inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setPrecision(Precision::U8);
        inputInfoFirst->getInputData()->setLayout(Layout::NCHW);

        OutputsDataMap outputInfo(net_reader.getNetwork().getOutputsInfo());

        for (auto&& item : outputInfo) {
            item.second->setPrecision(Precision::FP32);
            item.second->setLayout(InferenceEngine::TensorDesc::getLayoutByDims(item.second->getDims()));
        }

        input_name_ = inputInfo.begin()->first;
        net_ = config_.ie.LoadNetwork(net_reader.getNetwork(), config_.deviceName);
    }
}

std::vector<int> ieSizeToVector(const SizeVector& ie_output_dims) {
    std::vector<int> blob_sizes(ie_output_dims.size(), 0);
    for (size_t i = 0; i < blob_sizes.size(); ++i) {
        blob_sizes[i] = ie_output_dims[i];
    }
    return blob_sizes;
}

void ActionDetection::fetchResults() {
    if (!enabled()) return;
    results.clear();
    if (results_fetched_) return;
    results_fetched_ = true;

    const cv::Mat priorbox_out(ieSizeToVector(request->GetBlob(config_.priorbox_blob_name)->getTensorDesc().getDims()),
                               CV_32F, request->GetBlob(config_.priorbox_blob_name)->buffer());

    const cv::Mat loc_out(ieSizeToVector(request->GetBlob(config_.loc_blob_name)->getTensorDesc().getDims()),
                          CV_32F, request->GetBlob(config_.loc_blob_name)->buffer());

    const cv::Mat main_conf_out(ieSizeToVector(request->GetBlob(config_.detection_conf_blob_name)->getTensorDesc().getDims()),
                                CV_32F, request->GetBlob(config_.detection_conf_blob_name)->buffer());

    std::vector<cv::Mat> add_conf_out;
    for (int i = 0; i < config_.num_anchors; ++i) {
        const auto blob_name = config_.action_conf_blob_name_prefix + std::to_string(i + 1);
        add_conf_out.emplace_back(ieSizeToVector(request->GetBlob(blob_name)->getTensorDesc().getDims()),
                                  CV_32F, request->GetBlob(blob_name)->buffer());
    }

    /** Parse detections **/
    GetDetections(loc_out, main_conf_out, priorbox_out, add_conf_out,
                  cv::Size(static_cast<int>(width_), static_cast<int>(height_)), &results);
}

inline ActionDetection::NormalizedBBox
ActionDetection::ParseBBoxRecord(
        const float* data) const {
    NormalizedBBox bbox;
    bbox.xmin = data[0];
    bbox.ymin = data[1];
    bbox.xmax = data[2];
    bbox.ymax = data[3];
    return bbox;
}

cv::Rect ActionDetection::ConvertToRect(
        const NormalizedBBox& prior_bbox, const NormalizedBBox& variances,
        const NormalizedBBox& encoded_bbox, const cv::Size& frame_size) const {
    /** Convert prior bbox to CV_Rect **/
    const float prior_width = prior_bbox.xmax - prior_bbox.xmin;
    const float prior_height = prior_bbox.ymax - prior_bbox.ymin;
    const float prior_center_x = (prior_bbox.xmin + prior_bbox.xmax) / 2.0f;
    const float prior_center_y = (prior_bbox.ymin + prior_bbox.ymax) / 2.0f;

    /** Decode bbox coordinates from the SSD format **/
    const float decoded_bbox_center_x =
            variances.xmin * encoded_bbox.xmin * prior_width + prior_center_x;
    const float decoded_bbox_center_y =
            variances.ymin * encoded_bbox.ymin * prior_height + prior_center_y;
    const float decoded_bbox_width =
            static_cast<float>(exp(static_cast<float>(variances.xmax * encoded_bbox.xmax))) * prior_width;
    const float decoded_bbox_height =
            static_cast<float>(exp(static_cast<float>(variances.ymax * encoded_bbox.ymax))) * prior_height;

    /** Create decoded bbox **/
    NormalizedBBox decoded_bbox;
    decoded_bbox.xmin = decoded_bbox_center_x - decoded_bbox_width / 2.0f;
    decoded_bbox.ymin = decoded_bbox_center_y - decoded_bbox_height / 2.0f;
    decoded_bbox.xmax = decoded_bbox_center_x + decoded_bbox_width / 2.0f;
    decoded_bbox.ymax = decoded_bbox_center_y + decoded_bbox_height / 2.0f;

    /** Convert decoded bbox to CV_Rect **/
    return cv::Rect(static_cast<int>(decoded_bbox.xmin * frame_size.width),
                    static_cast<int>(decoded_bbox.ymin * frame_size.height),
                    static_cast<int>((decoded_bbox.xmax - decoded_bbox.xmin) * frame_size.width),
                    static_cast<int>((decoded_bbox.ymax - decoded_bbox.ymin) * frame_size.height));
}

void ActionDetection::GetDetections(const cv::Mat& loc, const cv::Mat& main_conf,
        const cv::Mat& priorbox, const std::vector<cv::Mat>& add_conf,
        const cv::Size& frame_size, DetectedActions* detections) const {
    /** num_candidates = H*W*NUM_SSD_ANCHORS **/
    const int num_candidates = priorbox.size[2] / SSD_PRIORBOX_RECORD_SIZE;

    /** Prepare input data buffers **/
    const float* loc_data = reinterpret_cast<float*>(loc.data);
    const float* det_conf_data = reinterpret_cast<float*>(main_conf.data);
    const float* prior_data = reinterpret_cast<float*>(priorbox.data);

    const int num_anchors = add_conf.size();
    std::vector<float*> action_conf_data(num_anchors);
    for (int i = 0; i < num_anchors; ++i) {
        action_conf_data[i] = reinterpret_cast<float*>(add_conf[i].data);
    }

    /** Variable to store all detection candidates**/
    DetectedActions valid_detections;

    /** Iterate over all candidate bboxes**/
    for (int p = 0; p < num_candidates; ++p) {
        /** Parse detection confidence from the SSD Detection output **/
        const float detection_conf =
                det_conf_data[p * NUM_DETECTION_CLASSES + POSITIVE_DETECTION_IDX];

        /** Skip low-confidence detections **/
        if (detection_conf < config_.detection_confidence_threshold) {
            continue;
        }

        /** Estimate the action label **/
        const int achor_id = p % num_anchors;
        const float* anchor_conf_data = action_conf_data[achor_id];
        const int action_conf_start_idx = p / num_anchors * config_.num_action_classes;
        int action_label = -1;
        float action_max_exp_value = 0.f;
        float action_sum_exp_values = 0.f;
        for (int c = 0; c < config_.num_action_classes; ++c) {
            float action_exp_value =
                std::exp(config_.action_scale * anchor_conf_data[action_conf_start_idx + c]);
            action_sum_exp_values += action_exp_value;
            if (action_exp_value > action_max_exp_value) {
                action_max_exp_value = action_exp_value;
                action_label = c;
            }
        }

        if (std::fabs(action_sum_exp_values) < std::numeric_limits<float>::epsilon()) {
            throw std::logic_error("action_sum_exp_values can't be equal to 0");
        }
        /** Estimate the action confidence **/
        float action_conf = action_max_exp_value / action_sum_exp_values;

        /** Skip low-confidence actions **/
        if (action_label < 0 || action_conf < config_.action_confidence_threshold) {
            action_label = config_.default_action_id;
            action_conf = 0.f;
        }

        /** Parse bbox from the SSD Detection output **/
        const auto priorbox =
                ParseBBoxRecord(prior_data + p * SSD_PRIORBOX_RECORD_SIZE);
        const auto variance =
                ParseBBoxRecord(prior_data + (num_candidates + p) * SSD_PRIORBOX_RECORD_SIZE);
        const auto encoded_bbox =
                ParseBBoxRecord(loc_data + p * SSD_LOCATION_RECORD_SIZE);

        const auto det_rect = ConvertToRect(priorbox, variance, encoded_bbox, frame_size);

        /** Store detected action **/
        valid_detections.emplace_back(det_rect, action_label, detection_conf, action_conf);
    }

    /** Merge most overlapped detections **/
    std::vector<int> out_det_indices;
    SoftNonMaxSuppression(valid_detections, config_.nms_sigma, config_.keep_top_k,
                          config_.detection_confidence_threshold,
                          &out_det_indices);

    detections->clear();
    for (size_t i = 0; i < out_det_indices.size(); ++i) {
        detections->emplace_back(valid_detections[out_det_indices[i]]);
    }
}

void ActionDetection::SoftNonMaxSuppression(const DetectedActions& detections,
        const float sigma, const int top_k, const float min_det_conf,
        std::vector<int>* out_indices) const {
    /** Store input bbox scores **/
    std::vector<float> scores(detections.size());
    for (size_t i = 0; i < detections.size(); ++i) {
        scores[i] = detections[i].detection_conf;
    }

    /** Estimate maximum number of algorithm iterations **/
    size_t max_num_steps = top_k > INVALID_TOP_K_IDX
                               ? std::min(static_cast<size_t>(top_k), scores.size())
                               : scores.size();

    /** Carry out Soft Non-Maximum Suppression algorithm **/
    out_indices->clear();
    for (size_t step = 0; step < max_num_steps; ++step) {
        auto best_score_itr = std::max_element(scores.begin(), scores.end());
        if (*best_score_itr < min_det_conf) {
            break;
        }

        /** Add current bbox to output list **/
        const int anchor_idx = static_cast<int>(std::distance(scores.begin(), best_score_itr));
        out_indices->emplace_back(anchor_idx);
        *best_score_itr = 0.f;

        /** Update scores of the rest bboxes **/
        for (size_t reference_idx = 0; reference_idx < scores.size(); ++reference_idx) {
            /** Skip updating step for the low-confidence bbox **/
            if (scores[reference_idx] < min_det_conf) {
                continue;
            }

            /** Calculate the Intersection over Union metric between two bboxes**/
            const auto& rect1 = detections[anchor_idx].rect;
            const auto& rect2 = detections[reference_idx].rect;
            const auto intersection = rect1 & rect2;
            float overlap = 0.f;
            if (intersection.width > 0 && intersection.height > 0) {
                const int intersection_area = intersection.area();
                overlap = static_cast<float>(intersection_area) / static_cast<float>(rect1.area() + rect2.area() - intersection_area);
            }

            /** Scale bbox score using the exponential rule **/
            scores[reference_idx] *= std::exp(-overlap * overlap / sigma);
        }
    }
}
