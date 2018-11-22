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

#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

#include "cnn.hpp"

/**
* @brief Class for detection with action info
*/
struct DetectedAction {
    /** @brief BBox of detection */
    cv::Rect rect;
    /** @brief Action label */
    int label;
    /** @brief Confidence of detection */
    float detection_conf;
    /** @brief Confidence of predicted action */
    float action_conf;

    /**
  * @brief Constructor
  */
    DetectedAction(const cv::Rect& rect, int label,
                   float detection_conf, float action_conf)
        : rect(rect), label(label), detection_conf(detection_conf),
          action_conf(action_conf) {}
};
using DetectedActions = std::vector<DetectedAction>;

/**
* @brief Config for the Action Detection model
*/
struct ActionDetectorConfig : public CnnConfig {
    explicit ActionDetectorConfig(const std::string& path_to_model,
                                  const std::string& path_to_weights)
        : CnnConfig(path_to_model, path_to_weights) {}
    /** @brief Name of output blob with location info */
    std::string loc_blob_name{"mbox_loc1/out/conv/flat"};
    /** @brief Name of output blob with detection confidence info */
    std::string detection_conf_blob_name{"mbox_main_conf/out/conv/flat/softmax/flat"};
    /** @brief Prefix of name of output blob with action confidence info */
    std::string action_conf_blob_name_prefix{"out/anchor"};
    /** @brief Name of output blob with priorbox info */
    std::string priorbox_blob_name{"mbox/priorbox"};
    /** @brief BBox overlap threshold for NMS algorithm */
    float nms_threshold = 0.45f;
    /** @brief Number of top-score bboxes for NMS algorithm */
    int nms_top_k = 400;
    /** @brief Threshold for detected objects */
    float detection_confidence_threshold = 0.65;
    /** @brief Number of top-score bboxes in output */
    int keep_top_k = 200;
    /** @brief Number of SSD anchors */
    int num_anchors = 4;
    /** @brief Number of actions to detect */
    int num_action_classes = 3;
    /** @brief Async execution flag */
    bool is_async = true;
};


class ActionDetection : public BaseCnnDetection {
public:
    explicit ActionDetection(const ActionDetectorConfig& config);

    DetectedActions results;
    void submitRequest() override;
    void enqueue(const cv::Mat &frame);
    void fetchResults();

private:
    ActionDetectorConfig config_;
    InferenceEngine::ExecutableNetwork net_;
    std::string input_name_;
    InferenceEngine::BlobMap outputs_;

    int enqueued_frames_ = 0;
    float width_ = 0;
    float height_ = 0;
    bool results_fetched_ = false;
    /**
    * @brief BBox in normalized form (each coordinate is in range [0;1]).
    */
    struct NormalizedBBox {
        float xmin;
        float ymin;
        float xmax;
        float ymax;
    };
    typedef std::vector<NormalizedBBox> NormalizedBBoxes;

     /**
    * @brief Translates the detections from the network outputs
    *
    * @param loc Location buffer
    * @param main_conf Detection conf buffer
    * @param add_conf Action conf buffer
    * @param priorbox Priorboxes buffer
    * @param frame_size Size of input image (WxH)
    * @param detections Detected objects
    */
    void GetDetections(const cv::Mat& loc,
                       const cv::Mat& main_conf,
                       const cv::Mat& priorbox,
                       const std::vector<cv::Mat>& add_conf,
                       const cv::Size& frame_size,
                       DetectedActions* detections) const;

     /**
    * @brief Translate input buffer to BBox
    *
    * @param data Input buffer
    * @return BBox
    */
    inline NormalizedBBox
    ParseBBoxRecord(const float* data) const;

     /**
    * @brief Translates input blobs in SSD format to bbox in CV_Rect
    *
    * @param prior_bbox Prior boxes in SSD format
    * @param variances Variances of prior boxes in SSD format
    * @param encoded_bbox BBox to decode
    * @param frame_size Size of input image (WxH)
    * @return BBox in CV_Rect format
    */
    cv::Rect ConvertToRect(const NormalizedBBox& prior_bbox,
                           const NormalizedBBox& variances,
                           const NormalizedBBox& encoded_bbox,
                           const cv::Size& frame_size) const;

     /**
    * @brief Carry out Non-Maximum Suppression algorithm under detected actions
    *
    * @param detections Detected actions
    * @param overlap_threshold Threshold to merge pair of bboxes
    * @param top_k Number of top-score bboxes
    * @param out_indices Out indices of valid detections
    */
    void NonMaxSuppression(const DetectedActions& detections,
                           const float overlap_threshold,
                           const int top_k,
                           std::vector<int>* out_indices) const;
};
