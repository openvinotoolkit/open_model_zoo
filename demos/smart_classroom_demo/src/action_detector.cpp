// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "action_detector.hpp"
#include <algorithm>
#include <utility>
#include <vector>
#include <limits>
#include <numeric>
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

        network_input_size_.height = inputInfoFirst->getTensorDesc().getDims()[2];
        network_input_size_.width = inputInfoFirst->getTensorDesc().getDims()[3];

        OutputsDataMap outputInfo(net_reader.getNetwork().getOutputsInfo());

        for (auto&& item : outputInfo) {
            item.second->setPrecision(Precision::FP32);
            item.second->setLayout(InferenceEngine::TensorDesc::getLayoutByDims(item.second->getDims()));
        }

        new_network_ = outputInfo.find(config_.new_loc_blob_name) != outputInfo.end();
        input_name_ = inputInfo.begin()->first;
        net_ = config_.ie.LoadNetwork(net_reader.getNetwork(), config_.deviceName);

        const auto& head_anchors = new_network_ ? config_.new_anchors : config_.old_anchors;
        const int num_heads = head_anchors.size();

        head_ranges_.resize(num_heads + 1);
        glob_anchor_map_.resize(num_heads);
        head_step_sizes_.resize(num_heads);

        num_glob_anchors_ = 0;
        head_ranges_[0] = 0;
        int head_shift = 0;
        for (int head_id = 0; head_id < num_heads; ++head_id) {
            glob_anchor_map_[head_id].resize(head_anchors[head_id]);

            int anchor_height, anchor_width;
            for (int anchor_id = 0; anchor_id < head_anchors[head_id]; ++anchor_id) {
                const auto glob_anchor_name = new_network_
                      ? config_.new_action_conf_blob_name_prefix + std::to_string(head_id + 1) +
                        config_.new_action_conf_blob_name_suffix + std::to_string(anchor_id + 1)
                      : config_.old_action_conf_blob_name_prefix + std::to_string(anchor_id + 1);
                glob_anchor_names_.push_back(glob_anchor_name);

                const auto anchor_dims = outputInfo[glob_anchor_name]->getDims();
                anchor_height = new_network_ ? anchor_dims[2] : anchor_dims[1];
                anchor_width = new_network_ ? anchor_dims[3] : anchor_dims[2];

                const int anchor_size = anchor_height * anchor_width;
                head_shift += anchor_size;

                head_step_sizes_[head_id] = new_network_ ? anchor_size : 1;
                glob_anchor_map_[head_id][anchor_id] = num_glob_anchors_++;
            }

            head_ranges_[head_id + 1] = head_shift;
            head_blob_sizes_.emplace_back(anchor_width, anchor_height);
        }

        num_candidates_ = head_shift;

        binary_task_ = config_.num_action_classes == 2;
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


    const auto loc_blob_name = new_network_ ? config_.new_loc_blob_name : config_.old_loc_blob_name;
    const auto det_conf_blob_name = new_network_ ? config_.new_det_conf_blob_name : config_.old_det_conf_blob_name;

    const cv::Mat priorbox_out =
        new_network_
          ? cv::Mat()
          : cv::Mat(ieSizeToVector(request->GetBlob(config_.old_priorbox_blob_name)->getTensorDesc().getDims()),
                    CV_32F, request->GetBlob(config_.old_priorbox_blob_name)->buffer());

    const cv::Mat loc_out(ieSizeToVector(request->GetBlob(loc_blob_name)->getTensorDesc().getDims()),
                          CV_32F, request->GetBlob(loc_blob_name)->buffer());

    const cv::Mat main_conf_out(ieSizeToVector(request->GetBlob(det_conf_blob_name)->getTensorDesc().getDims()),
                                CV_32F, request->GetBlob(det_conf_blob_name)->buffer());

    std::vector<cv::Mat> add_conf_out;
    for (int glob_anchor_id = 0; glob_anchor_id < num_glob_anchors_; ++glob_anchor_id) {
        const auto& blob_name = glob_anchor_names_[glob_anchor_id];
        add_conf_out.emplace_back(ieSizeToVector(request->GetBlob(blob_name)->getTensorDesc().getDims()),
                                  CV_32F, request->GetBlob(blob_name)->buffer());
    }

    /** Parse detections **/
    GetDetections(loc_out, main_conf_out, priorbox_out, add_conf_out,
                  cv::Size(static_cast<int>(width_), static_cast<int>(height_)), &results);
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
    const float row = pos / blob_size.width;
    const float col = pos % blob_size.width;

    const float center_x = (col + 0.5f) * static_cast<float>(step);
    const float center_y = (row + 0.5f) * static_cast<float>(step);

    NormalizedBBox bbox;
    bbox.xmin = (center_x - 0.5f * anchor.width) / static_cast<float>(network_input_size_.width);
    bbox.ymin = (center_y - 0.5f * anchor.height) / static_cast<float>(network_input_size_.height);
    bbox.xmax = (center_x + 0.5f * anchor.width) / static_cast<float>(network_input_size_.width);
    bbox.ymax = (center_y + 0.5f * anchor.height) / static_cast<float>(network_input_size_.height);

    return bbox;
}

cv::Rect ActionDetection::ConvertToRect(
        const NormalizedBBox& prior_bbox, const NormalizedBBox& variances,
        const NormalizedBBox& encoded_bbox, const cv::Size& frame_size) const {
    /** Convert prior bbox to CV_Rect **/
    const float prior_width = prior_bbox.xmax - prior_bbox.xmin;
    const float prior_height = prior_bbox.ymax - prior_bbox.ymin;
    const float prior_center_x = 0.5f * (prior_bbox.xmin + prior_bbox.xmax);
    const float prior_center_y = 0.5f * (prior_bbox.ymin + prior_bbox.ymax);

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
    const float decoded_bbox_xmin = decoded_bbox_center_x - 0.5f * decoded_bbox_width;
    const float decoded_bbox_ymin = decoded_bbox_center_y - 0.5f * decoded_bbox_height;
    const float decoded_bbox_xmax = decoded_bbox_center_x + 0.5f * decoded_bbox_width;
    const float decoded_bbox_ymax = decoded_bbox_center_y + 0.5f * decoded_bbox_height;

    /** Convert decoded bbox to CV_Rect **/
    return cv::Rect(static_cast<int>(decoded_bbox_xmin * frame_size.width),
                    static_cast<int>(decoded_bbox_ymin * frame_size.height),
                    static_cast<int>((decoded_bbox_xmax - decoded_bbox_xmin) * frame_size.width),
                    static_cast<int>((decoded_bbox_ymax - decoded_bbox_ymin) * frame_size.height));
}

void ActionDetection::GetDetections(const cv::Mat& loc, const cv::Mat& main_conf,
        const cv::Mat& priorboxes, const std::vector<cv::Mat>& add_conf,
        const cv::Size& frame_size, DetectedActions* detections) const {
    /** Prepare input data buffers **/
    const float* loc_data = reinterpret_cast<float*>(loc.data);
    const float* det_conf_data = reinterpret_cast<float*>(main_conf.data);
    const float* prior_data = new_network_ ? NULL : reinterpret_cast<float*>(priorboxes.data);

    const int total_num_anchors = add_conf.size();
    std::vector<float*> action_conf_data(total_num_anchors);
    for (int i = 0; i < total_num_anchors; ++i) {
        action_conf_data[i] = reinterpret_cast<float*>(add_conf[i].data);
    }

    /** Variable to store all detection candidates**/
    DetectedActions valid_detections;

    /** Iterate over all candidate bboxes**/
    for (int p = 0; p < num_candidates_; ++p) {
        /** Parse detection confidence from the SSD Detection output **/
        const float detection_conf =
                det_conf_data[p * NUM_DETECTION_CLASSES + POSITIVE_DETECTION_IDX];

        /** Skip low-confidence detections **/
        if (detection_conf < config_.detection_confidence_threshold) {
            continue;
        }

        /** Estimate the action head ID **/
        int head_id = 0;
        while (p >= head_ranges_[head_id + 1]) {
            ++head_id;
        }
        const int head_p = p - head_ranges_[head_id];

        /** Estimate the action anchor ID **/
        const int head_num_anchors =
            new_network_ ? config_.new_anchors[head_id] : config_.old_anchors[head_id];
        const int anchor_id = head_p % head_num_anchors;

        /** Estimate the action label **/
        const int glob_anchor_id = glob_anchor_map_[head_id][anchor_id];
        const float* anchor_conf_data = action_conf_data[glob_anchor_id];
        const int action_conf_idx_shift = new_network_
                                            ? head_p / head_num_anchors
                                            : head_p / head_num_anchors * config_.num_action_classes;
        const int action_conf_step = head_step_sizes_[head_id];
        const float scale = new_network_ ? config_.new_action_scale : config_.old_action_scale;
        int action_label = -1;
        float action_max_exp_value = 0.f;
        float action_sum_exp_values = 0.f;
        for (int c = 0; c < config_.num_action_classes; ++c) {
            float action_exp_value =
                std::exp(scale * anchor_conf_data[action_conf_idx_shift + c * action_conf_step]);
            action_sum_exp_values += action_exp_value;
            if (action_exp_value > action_max_exp_value && ((c > 0 && binary_task_) || !binary_task_)) {
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
        const auto priorbox = new_network_
                                ? GeneratePriorBox(head_p / head_num_anchors,
                                                   config_.new_det_heads[head_id].step,
                                                   config_.new_det_heads[head_id].anchors[anchor_id],
                                                   head_blob_sizes_[head_id])
                                : ParseBBoxRecord(prior_data + p * SSD_PRIORBOX_RECORD_SIZE, false);
        const auto variance =
                ParseBBoxRecord(new_network_
                                    ? config_.variances
                                    : prior_data + (num_candidates_ + p) * SSD_PRIORBOX_RECORD_SIZE,
                                false);
        const auto encoded_bbox =
                ParseBBoxRecord(loc_data + p * SSD_LOCATION_RECORD_SIZE, new_network_);

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
    size_t max_queue_size = top_k > INVALID_TOP_K_IDX
                               ? std::min(static_cast<size_t>(top_k), scores.size())
                               : scores.size();

    /** Select top-k score indices **/
    std::vector<size_t> score_idx(scores.size());
    std::iota(score_idx.begin(), score_idx.end(), 0);
    std::partial_sort(score_idx.begin(), score_idx.begin() + max_queue_size, score_idx.end(),
        [&scores](size_t i1, size_t i2) {return scores[i1] > scores[i2];});

    /** Extract top-k score values **/
    std::vector<size_t> valid_score_idx(score_idx.begin(), score_idx.begin() + max_queue_size);
    std::vector<float> valid_scores(max_queue_size);
    for (size_t i = 0; i < valid_score_idx.size(); ++i) {
        valid_scores[i] = scores[valid_score_idx[i]];
    }

    /** Carry out Soft Non-Maximum Suppression algorithm **/
    out_indices->clear();
    for (size_t step = 0; step < valid_scores.size(); ++step) {
        auto best_score_itr = std::max_element(valid_scores.begin(), valid_scores.end());
        if (*best_score_itr < min_det_conf) {
            break;
        }

        /** Add current bbox to output list **/
        const size_t local_anchor_idx = std::distance(valid_scores.begin(), best_score_itr);
        const int anchor_idx = valid_score_idx[local_anchor_idx];
        out_indices->emplace_back(anchor_idx);
        *best_score_itr = 0.f;

        /** Update valid_scores of the rest bboxes **/
        for (size_t local_reference_idx = 0; local_reference_idx < valid_scores.size(); ++local_reference_idx) {
            /** Skip updating step for the low-confidence bbox **/
            if (valid_scores[local_reference_idx] < min_det_conf) {
                continue;
            }

            /** Calculate the Intersection over Union metric between two bboxes**/
            const size_t reference_idx = valid_score_idx[local_reference_idx];
            const auto& rect1 = detections[anchor_idx].rect;
            const auto& rect2 = detections[reference_idx].rect;
            const auto intersection = rect1 & rect2;
            float overlap = 0.f;
            if (intersection.width > 0 && intersection.height > 0) {
                const int intersection_area = intersection.area();
                overlap = static_cast<float>(intersection_area) / static_cast<float>(rect1.area() + rect2.area() - intersection_area);
            }

            /** Scale bbox score using the exponential rule **/
            valid_scores[local_reference_idx] *= std::exp(-overlap * overlap / sigma);
        }
    }
}
