// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom_kernels.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <vector>

#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/imgproc.hpp>

#include "tracker.hpp"

const float BOUNDING_BOX_THRESHOLD = 0.4f;
const int ACTION_IMAGE_SCALE = 256;

namespace {
cv::Rect convert_to_central_roi(const cv::Rect& person_roi, const cv::Size& in_size, const size_t scale) {
    const int roi_height = person_roi.height;
    const int roi_width = person_roi.width;

    const int src_roi_center_x = static_cast<int>(0.5 * (person_roi.tl().x + person_roi.br().x));
    const int src_roi_center_y = static_cast<int>(0.5 * (person_roi.tl().y + person_roi.br().y));

    const float height_scale = static_cast<float>(in_size.height) / static_cast<float>(scale);
    const float width_scale = static_cast<float>(in_size.width) / static_cast<float>(scale);

    CV_DbgAssert(height_scale < 1.0f);
    CV_DbgAssert(width_scale < 1.0f);

    const int min_roi_size = std::min(roi_height, roi_width);
    const int trg_roi_height = static_cast<int>(height_scale * min_roi_size);
    const int trg_roi_width = static_cast<int>(width_scale * min_roi_size);

    return cv::Rect(static_cast<int>(src_roi_center_x - 0.5 * trg_roi_width),
                    static_cast<int>(src_roi_center_y - 0.5 * trg_roi_height),
                    trg_roi_width,
                    trg_roi_height);
}
}  // anonymous namespace

// clang-format off
GAPI_OCV_KERNEL(OCVGetFastFrame, custom::GetFastFrame) {
    static void run(const std::vector<cv::Mat>& batch,
                    const cv::Size& image_size,
                          cv::Mat& frame) {
        const uint8_t* ptrI = batch[batch.size() - 2].ptr<uint8_t>();
        std::copy(ptrI, ptrI + image_size.area() * 3, frame.ptr<uint8_t>());
    }
};

GAPI_OCV_KERNEL(OCVExtractBoundingBox, custom::ExtractBoundingBox) {
    static void run(const cv::Mat& in_ssd_result,
                    const cv::Mat& in_frame,
                    const cv::Scalar& net_size,
                          TrackedObjects& detections) {
        float scaling_x = static_cast<float>(in_frame.size().width / net_size[3]);
        float scaling_y = static_cast<float>(in_frame.size().height / net_size[2]);

        detections.clear();
        const float *data = in_ssd_result.ptr<float>();
        for (int i = 0; i < 100; i++) {
            const int OBJECT_SIZE = 5;

            const float x_min = data[i * OBJECT_SIZE + 0];
            const float y_min = data[i * OBJECT_SIZE + 1];
            const float x_max = data[i * OBJECT_SIZE + 2];
            const float y_max = data[i * OBJECT_SIZE + 3];
            const float conf  = data[i * OBJECT_SIZE + 4];

            if (conf > BOUNDING_BOX_THRESHOLD) {
                TrackedObject object;
                object.rect = cv::Rect(static_cast<int>(x_min * scaling_x),
                                       static_cast<int>(y_min * scaling_y),
                                       static_cast<int>((x_max - x_min) * scaling_x),
                                       static_cast<int>((y_max - y_min) * scaling_y));
                object.confidence = conf;
                detections.push_back(object);
            }
        }
    }
};

GAPI_OCV_KERNEL_ST(OCVTrackPerson, custom::TrackPerson, Tracker) {
    static void setup(const cv::GMatDesc&,
                      const cv::GArrayDesc&,
                            std::shared_ptr<Tracker>& tracker_action,
                      const cv::GCompileArgs& compileArgs) {
        TrackerParams tracker_action_params;
        tracker_action_params.min_track_duration = 20;
        tracker_action_params.forget_delay = 150;
        tracker_action_params.affinity_thr = 0.9f;
        tracker_action_params.averaging_window_size_for_rects = 5;
        tracker_action_params.averaging_window_size_for_labels = 1;
        tracker_action_params.drop_forgotten_tracks = true;
        tracker_action_params.max_num_objects_in_track = 5;
        tracker_action = std::make_shared<Tracker>(tracker_action_params);
    }
    static void run(const cv::Mat& frame,
                    const TrackedObjects& actions,
                          TrackedObjects& tracked_actions,
                          Tracker& tracker_action) {
        tracker_action.process(frame, actions);
        tracked_actions = tracker_action.trackedDetectionsWithLabels();
    }
};
// clang-format on
namespace BatchState {
struct Params {
    cv::Mat current_frame;
    cv::Mat prepared_mat;
    std::atomic<size_t> last_id;
    Params() {}
    Params(const Params& params) {
        current_frame = params.current_frame;
        prepared_mat = params.prepared_mat;
        last_id.fetch_add(params.last_id);
    }
    Params& operator=(const Params& params) {
        current_frame = params.current_frame;
        prepared_mat = params.prepared_mat;
        last_id.fetch_add(params.last_id);
        return *this;
    }
};
}  // namespace BatchState

// clang-format off
GAPI_OCV_KERNEL_ST(OCVConstructClip, custom::ConstructClip, BatchState::Params) {
    static void setup(const cv::GArrayDesc&,
                      const cv::GArrayDesc&,
                      const cv::Scalar& net_size,
                      const cv::Size& image_size,
                      const cv::GOpaqueDesc&,
                            std::shared_ptr<BatchState::Params>& state) {
        BatchState::Params params;
        params.last_id = 0;
        params.prepared_mat.create(std::vector<int>{1,
                                                    static_cast<int>(net_size[0]),
                                                    static_cast<int>(net_size[1]),
                                                    static_cast<int>(net_size[2]),
                                                    static_cast<int>(net_size[3])}, CV_32F);
        state = std::make_shared<BatchState::Params>(params);
    }
    static void run(const std::vector<cv::Mat>& batch,
                    const TrackedObjects& tracked_persons,
                    const cv::Scalar& net_size,
                    const cv::Size& image_size,
                    const std::shared_ptr<size_t>& current_person_id,
                          std::vector<cv::Mat>& wrapped_mat,
                    BatchState::Params& state) {
        const int duration = static_cast<int>(net_size[1]);
        const int height = static_cast<int>(net_size[2]);
        const int width = static_cast<int>(net_size[3]);
        const auto ptr = batch[batch.size() - 1].ptr<uint8_t>();
        auto p_pm = state.prepared_mat.ptr<float>();

        if (ptr[1] > 0) {  // is filled and updated
            int step = ptr[0];  // first
            size_t& person_id = *current_person_id;
            if (person_id < tracked_persons.size()) {  // wrong number protection
                state.last_id = person_id;
            }
            for (int i = 0; i < duration; ++i) {
                cv::Mat current_frame = batch[step];
                if (++step > (duration - 1)) {
                    step = 0;
                }

                if (tracked_persons.size() > 0) {
                    cv::Mat crop, cvt, resized;
                    cv::Rect roi = tracked_persons.at(state.last_id).rect;

                    crop = current_frame(convert_to_central_roi(roi,
                                                                cv::Size(height, width),
                                                                ACTION_IMAGE_SCALE));

                    crop.convertTo(cvt, CV_32F);
                    cv::resize(cvt, resized, cv::Size{height, width});

                    cv::Mat different_channels[3];
                    cv::split(resized, different_channels);

                    std::vector<float*> rgb = {different_channels[2].ptr<float>(),
                                               different_channels[1].ptr<float>(),
                                               different_channels[0].ptr<float>()};

                    for (int ch = 0; ch < 3; ++ch) {
                        std::copy(rgb[ch], rgb[ch] + height * width,
                                  p_pm + ch * duration * height * width
                                      + i * height * width);
                    }
                }
            }
        }
        wrapped_mat.push_back(state.prepared_mat);
    }
};

GAPI_OCV_KERNEL(OCVGestureRecognitionPostprocessing, custom::GestureRecognitionPostprocessing) {
    static void run(const std::vector<cv::Mat>& asl_result,
                    const float ar_threshold,
                          int& label_number) {
        label_number = -1;
        if (!asl_result.empty()) {
            double min = 0., max = 0.;
            int minIdx = 0, maxIdx = 0;
            const float* data = asl_result[0].ptr<float>();
            // Find more suitable action
            cv::minMaxIdx(asl_result[0].reshape(1,
                static_cast<int>(asl_result[0].total())), &min, &max, &minIdx, &maxIdx);
            if (data[maxIdx] > ar_threshold) {
                label_number = maxIdx;
            }
        }
    }
};
// clang-format on

cv::gapi::GKernelPackage custom::kernels() {
    return cv::gapi::kernels<OCVExtractBoundingBox,
                             OCVTrackPerson,
                             OCVConstructClip,
                             OCVGestureRecognitionPostprocessing,
                             OCVGetFastFrame>();
}
