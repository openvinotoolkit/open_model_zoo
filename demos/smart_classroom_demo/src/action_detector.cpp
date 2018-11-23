/*
// Copyright (c) 2018 Intel Corporation
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

#include "action_detector.hpp"
#include <utility>
#include <vector>
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

    width_ = frame.cols;
    height_ = frame.rows;

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
            item.second->precision = Precision::FP32;
            item.second->setLayout(InferenceEngine::TensorDesc::getLayoutByDims(item.second->getDims()));
        }

        input_name_ = inputInfo.begin()->first;
        net_ = config_.plugin.LoadNetwork(net_reader.getNetwork(), {});
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
                  cv::Size(width_, height_), &results);
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
    const float prior_center_x = (prior_bbox.xmin + prior_bbox.xmax) / 2.;
    const float prior_center_y = (prior_bbox.ymin + prior_bbox.ymax) / 2.;

    /** Decode bbox coordinates from the SSD format **/
    const float decoded_bbox_center_x =
            variances.xmin * encoded_bbox.xmin * prior_width + prior_center_x;
    const float decoded_bbox_center_y =
            variances.ymin * encoded_bbox.ymin * prior_height + prior_center_y;
    const float decoded_bbox_width =
            exp(variances.xmax * encoded_bbox.xmax) * prior_width;
    const float decoded_bbox_height =
            exp(variances.ymax * encoded_bbox.ymax) * prior_height;

    /** Create decoded bbox **/
    NormalizedBBox decoded_bbox;
    decoded_bbox.xmin = decoded_bbox_center_x - decoded_bbox_width / 2.;
    decoded_bbox.ymin = decoded_bbox_center_y - decoded_bbox_height / 2.;
    decoded_bbox.xmax = decoded_bbox_center_x + decoded_bbox_width / 2.;
    decoded_bbox.ymax = decoded_bbox_center_y + decoded_bbox_height / 2.;

    /** Convert decoded bbox to CV_Rect **/
    return cv::Rect(decoded_bbox.xmin * frame_size.width,
                    decoded_bbox.ymin * frame_size.height,
                    (decoded_bbox.xmax - decoded_bbox.xmin) * frame_size.width,
                    (decoded_bbox.ymax - decoded_bbox.ymin) * frame_size.height);
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
        int action_label = 0;
        float action_conf = anchor_conf_data[action_conf_start_idx];
        for (int c = 1; c < config_.num_action_classes; ++c) {
            if (anchor_conf_data[action_conf_start_idx + c] > action_conf) {
                action_conf = anchor_conf_data[action_conf_start_idx + c];
                action_label = c;
            }
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
    NonMaxSuppression(valid_detections, config_.nms_threshold, config_.nms_top_k,
                      &out_det_indices);

    /** Select keep_top_k detections with highest detection confidence if possible **/
    const int num_detections = out_det_indices.size();
    if (config_.keep_top_k > INVALID_TOP_K_IDX
            && num_detections > config_.keep_top_k) {
        /** Store input bbox scores with idx **/
        std::vector<std::pair<float, int> > indexed_scores;
        for (int i = 0; i < num_detections; ++i) {
            const int idx = out_det_indices[i];
            indexed_scores.emplace_back(valid_detections[idx].detection_conf, idx);
        }
        /** Sort indexed scores in descending order **/
        std::stable_sort(indexed_scores.begin(), indexed_scores.end(),
                         SortScorePairDescend<int>);
        /** Select top_k scores **/
        indexed_scores.resize(config_.keep_top_k);

        /** Store the selected indices **/
        std::vector<int> selected_indices;
        for (size_t i = 0; i < indexed_scores.size(); ++i) {
            selected_indices.push_back(indexed_scores[i].second);
        }

        out_det_indices.swap(selected_indices);
    }

    detections->clear();
    for (size_t i = 0; i < out_det_indices.size(); ++i) {
        detections->emplace_back(valid_detections[out_det_indices[i]]);
    }
}

void ActionDetection::NonMaxSuppression(
        const DetectedActions& detections,
        const float overlap_threshold, const int top_k,
        std::vector<int>* out_indices) const {
    /** Store input bbox scores with idx **/
    std::vector<std::pair<float, int> > indexed_scores;
    for (size_t i = 0; i < detections.size(); ++i) {
        indexed_scores.emplace_back(detections[i].detection_conf, i);
    }
    /** Sort indexed scores in descending order **/
    std::stable_sort(indexed_scores.begin(), indexed_scores.end(),
                     SortScorePairDescend<int>);
    /** Select top_k scores if possible **/
    if (top_k > INVALID_TOP_K_IDX
            && top_k < static_cast<int>(indexed_scores.size())) {
        indexed_scores.resize(top_k);
    }

    /** Carry out Non-Maximum Suppression algorithm **/
    out_indices->clear();
    for (const auto& item : indexed_scores) {
        const int& anchor_idx = item.second;

        bool keep_idx = true;
        for (int reference_idx : *out_indices) {
            /** Calculate the Intersection over Union metric between two bboxes**/
            const auto& rect1 = detections[anchor_idx].rect;
            const auto& rect2 = detections[reference_idx].rect;
            const auto intersection = rect1 & rect2;
            float overlap = 0.f;
            if (intersection.width > 0.f && intersection.height > 0.f) {
                const float intersection_area = intersection.area();
                overlap = intersection_area / (rect1.area() + rect2.area() - intersection_area);
            }

            /** Remove overlapped bbox with lowest confidence **/
            if (overlap > overlap_threshold) {
                keep_idx = false;
                break;
            }
        }

        /** Store output bbox **/
        if (keep_idx) {
            out_indices->emplace_back(anchor_idx);
        }
    }
}
