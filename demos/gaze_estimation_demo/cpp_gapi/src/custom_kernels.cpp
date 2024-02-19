// Copyright (C) 2021-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "custom_kernels.hpp"

#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <memory>

#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/imgproc.hpp>

#include "kernel_packages.hpp"

namespace {
void rotateImageAroundCenter(const cv::Mat& srcImage, const float angle, cv::Mat& dstImage) {
    const auto width = srcImage.cols;
    const auto height = srcImage.rows;

    const cv::Size size(width, height);

    const cv::Point2f center(static_cast<float>(width / 2), static_cast<float>(height / 2));

    const auto rotMatrix = cv::getRotationMatrix2D(center, static_cast<double>(angle), 1);
    cv::warpAffine(srcImage, dstImage, rotMatrix, size, 1, cv::BORDER_REPLICATE);
}

void toCHW(const cv::Mat& src, cv::Mat& dst) {
    dst.create(cv::Size(src.cols, src.rows * src.channels()), CV_32F);
    std::vector<cv::Mat> planes;
    for (int i = 0; i < src.channels(); ++i) {
        planes.push_back(dst.rowRange(i * src.rows, (i + 1) * src.rows));
    }
    cv::split(src, planes);
}

void preprocessing(const cv::Mat& src, const cv::Size& new_size, const float roll, cv::Mat& dst) {
    cv::Mat rot, cvt, rsz;
    rotateImageAroundCenter(src, roll, rot);  // rotate
    cv::resize(rot, rsz, new_size);  // resize
    rsz.convertTo(cvt, CV_32F);  // convert to F32
    toCHW(cvt, dst);  // HWC to CHW
    dst = dst.reshape(1, {1, 3, new_size.height, new_size.width});  // reshape to CNN input
}

void adjustBoundingBox(cv::Rect& boundingBox) {
    auto w = boundingBox.width;
    auto h = boundingBox.height;

    boundingBox.x -= static_cast<int>(0.067 * w);
    boundingBox.y -= static_cast<int>(0.028 * h);

    boundingBox.width += static_cast<int>(0.15 * w);
    boundingBox.height += static_cast<int>(0.13 * h);

    if (boundingBox.width < boundingBox.height) {
        auto dx = (boundingBox.height - boundingBox.width);
        boundingBox.x -= dx / 2;
        boundingBox.width += dx;
    } else {
        auto dy = (boundingBox.width - boundingBox.height);
        boundingBox.y -= dy / 2;
        boundingBox.height += dy;
    }
}

cv::Rect createEyeBoundingBox(const cv::Point2i& p1, const cv::Point2i& p2, float scale = 1.8f) {
    cv::Rect result;
    float size = static_cast<float>(cv::norm(p1 - p2));

    result.width = static_cast<int>(scale * size);
    result.height = result.width;

    auto midpoint = (p1 + p2) / 2;

    result.x = midpoint.x - (result.width / 2);
    result.y = midpoint.y - (result.height / 2);

    return result;
}
}  // anonymous namespace

// clang-format off
GAPI_OCV_KERNEL(OCVPrepareEyes, custom::PrepareEyes) {
    static void run(const cv::Mat& in,
                    const std::vector<cv::Rect>& left_rc,
                    const std::vector<cv::Rect>& right_rc,
                    const std::vector<cv::Mat>& rolls,
                    const cv::Size& new_size,
                          std::vector<cv::Mat>& left_eyes,
                          std::vector<cv::Mat>& right_eyes) {
        left_eyes.clear();
        right_eyes.clear();
        const auto size = left_rc.size();
        for (size_t j = 0; j < size; ++j) {
            auto roll = rolls.at(j).ptr<float>()[0];

            cv::Mat l_dst, r_dst;
            auto leftEyeImage(cv::Mat(in, left_rc.at(j)));
            preprocessing(leftEyeImage, new_size, roll, l_dst);
            left_eyes.push_back(l_dst);

            auto rightEyeImage(cv::Mat(in, right_rc.at(j)));
            preprocessing(rightEyeImage, new_size, roll, r_dst);
            right_eyes.push_back(r_dst);
        }
    }
};

/** FIXME: This kernel should become part of G-API kernels in future **/
GAPI_OCV_KERNEL(OCVParseSSD, custom::ParseSSD) {
    static void run(const cv::Mat& in_ssd_result,
                    const cv::Size& upscale,
                    const float detectionThreshold,
                          std::vector<cv::Rect>& out_objects,
                          std::vector<float>& out_confidence) {
        const auto &in_ssd_dims = in_ssd_result.size;
        CV_Assert(in_ssd_dims.dims() == 4u);

        const int MAX_PROPOSALS = in_ssd_dims[2];
        const int OBJECT_SIZE   = in_ssd_dims[3];
        CV_Assert(OBJECT_SIZE  == 7);  // fixed SSD object size

        const cv::Rect surface({0, 0}, upscale);
        out_objects.clear();

        const float *data = in_ssd_result.ptr<float>();
        for (int i = 0; i < MAX_PROPOSALS; i++) {
            const float image_id   = data[i * OBJECT_SIZE + 0];
            const float confidence = data[i * OBJECT_SIZE + 2];

            if (image_id < 0.f) {
                break;    // marks end-of-detections
            }
            if (confidence < detectionThreshold) {
                continue;  // skip objects with low confidence
            }
            const float rc_left    = data[i * OBJECT_SIZE + 3];
            const float rc_top     = data[i * OBJECT_SIZE + 4];
            const float rc_right   = data[i * OBJECT_SIZE + 5];
            const float rc_bottom  = data[i * OBJECT_SIZE + 6];
            out_confidence.push_back(confidence);
            cv::Rect rc;  // map relative coordinates to the original image scale
            rc.x      = static_cast<int>(rc_left   * upscale.width);
            rc.y      = static_cast<int>(rc_top    * upscale.height);
            rc.width  = static_cast<int>(rc_right  * upscale.width)  - rc.x;
            rc.height = static_cast<int>(rc_bottom * upscale.height) - rc.y;
            adjustBoundingBox(rc);

            const auto clipped_rc = rc & surface;
            if (clipped_rc.area() != rc.area()) {
                continue;
            }
            out_objects.emplace_back(clipped_rc);
        }
    }
};

GAPI_OCV_KERNEL(OCVProcessPoses, custom::ProcessPoses) {
    static void run(const std::vector<cv::Mat>& in_ys,
                    const std::vector<cv::Mat>& in_ps,
                    const std::vector<cv::Mat>& in_rs,
                          std::vector<cv::Point3f>& out_poses,
                          std::vector<cv::Mat>& out_poses_wr) {
        CV_Assert(in_ys.size() == in_ps.size() && in_ys.size() == in_rs.size());
        const size_t sz = in_ys.size();
        for (size_t idx = 0u; idx < sz; ++idx) {
            cv::Point3f pose;
            pose.x = in_ys[idx].ptr<float>()[0];
            pose.y = in_ps[idx].ptr<float>()[0];
            pose.z = in_rs[idx].ptr<float>()[0];
            out_poses.push_back(pose);

            cv::Mat pose_wr(1, 3, CV_32FC1);
            float* ptr = pose_wr.ptr<float>();
            ptr[0] = pose.x;
            ptr[1] = pose.y;
            ptr[2] = 0;
            out_poses_wr.push_back(pose_wr);
        }
    }
};

GAPI_OCV_KERNEL(OCVProcessLandmarks, custom::ProcessLandmarks) {
    static void run(const cv::Mat& in,
                    const std::vector<cv::Mat>& landmarks,
                    const std::vector<cv::Rect>& face_rois,
                          std::vector<cv::Rect>& left_eyes_rc,
                          std::vector<cv::Rect>& right_eyes_rc,
                          std::vector<cv::Point2f>& leftEyeMidpoint,
                          std::vector<cv::Point2f>& rightEyeMidpoint,
                          std::vector<std::vector<cv::Point>>& out_lanmarks) {
        CV_Assert(landmarks.size() == face_rois.size());
        const auto size = landmarks.size();
        for (size_t idx = 0u; idx < size; ++idx) {
            const float* rawLandmarks = landmarks.at(idx).ptr<float>();
            std::vector<cv::Point2i> faceLandmarks;

            for (unsigned long i = 0; i < landmarks.at(idx).total() / 2; ++i) {
                const int x = static_cast<int>(rawLandmarks[2 * i] *
                    face_rois.at(idx).width + face_rois.at(idx).x);
                const int y = static_cast<int>(rawLandmarks[2 * i + 1] *
                    face_rois.at(idx).height + face_rois.at(idx).y);
                faceLandmarks.emplace_back(x, y);
            }
            leftEyeMidpoint.push_back((faceLandmarks[0] + faceLandmarks[1]) / 2);
            left_eyes_rc.push_back(createEyeBoundingBox(faceLandmarks[0], faceLandmarks[1]));

            rightEyeMidpoint.push_back((faceLandmarks[2] + faceLandmarks[3]) / 2);
            right_eyes_rc.push_back(createEyeBoundingBox(faceLandmarks[2], faceLandmarks[3]));

            out_lanmarks.push_back(faceLandmarks);
        }
    }
};

GAPI_OCV_KERNEL(OCVProcessEyes, custom::ProcessEyes) {
    static void run(const cv::Mat& in,
                    const std::vector<cv::Mat>& left_eyes_state,
                    const std::vector<cv::Mat>& right_eyes_state,
                          std::vector<int>& left_states,
                          std::vector<int>& right_states) {
        CV_Assert(left_eyes_state.size() == right_eyes_state.size());
        for (const auto& state : left_eyes_state) {
            const auto left_outputValue = state.ptr<float>();
            const auto left = left_outputValue[0] < left_outputValue[1] ? 1 : 0;
            left_states.push_back(left);
        }

        for (const auto& state : right_eyes_state) {
            const auto right_outputValue = state.ptr<float>();
            const auto right = right_outputValue[0] < right_outputValue[1] ? 1 : 0;
            right_states.push_back(right);
        }
    }
};

GAPI_OCV_KERNEL(OCVProcessGazes, custom::ProcessGazes) {
    static void run(const std::vector<cv::Mat>& gaze_vectors,
                    const std::vector<cv::Mat>& rolls,
                          std::vector<cv::Point3f>& out_gazes) {
        CV_Assert(gaze_vectors.size() == rolls.size());
        const auto size = gaze_vectors.size();
        for (size_t i = 0; i < size; ++i) {
            const auto rawResults = gaze_vectors.at(i).ptr<float>();
            cv::Point3f gazeVector;
            gazeVector.x = rawResults[0];
            gazeVector.y = rawResults[1];
            gazeVector.z = rawResults[2];

            gazeVector = gazeVector / cv::norm(gazeVector);

            /** rotate gaze vector to compensate for the alignment **/
            auto roll = rolls.at(i).ptr<float>()[0];
            float cs = static_cast<float>(std::cos(static_cast<double>(roll) * CV_PI / 180.0));
            float sn = static_cast<float>(std::sin(static_cast<double>(roll) * CV_PI / 180.0));
            auto tmpX = gazeVector.x * cs + gazeVector.y * sn;
            auto tmpY = -gazeVector.x * sn + gazeVector.y * cs;
            gazeVector.x = tmpX;
            gazeVector.y = tmpY;

            out_gazes.push_back(gazeVector);
        }
    }
};
// clang-format on

cv::gapi::GKernelPackage custom::kernels() {
    return cv::gapi::
        kernels<OCVPrepareEyes, OCVParseSSD, OCVProcessLandmarks, OCVProcessEyes, OCVProcessGazes, OCVProcessPoses>();
}
