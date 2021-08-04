// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "custom_kernels.hpp"

#include <opencv2/imgproc.hpp>

namespace {
const float P_NET_WINDOW_SIZE = 12.0f;

std::vector<custom::Face> buildFaces(const cv::Mat& scores,
                             const cv::Mat& regressions,
                             const float scaleFactor,
                             const float threshold) {

    auto w = scores.size[3];
    auto h = scores.size[2];
    auto size = w * h;

    const float* scores_data = scores.ptr<float>();
    scores_data += size;

    const float* reg_data = regressions.ptr<float>();

    auto out_side = std::max(h, w);
    auto in_side = 2 * out_side + 11;
    float stride = 0.0f;
    if (out_side != 1) {
        stride = static_cast<float>(in_side - P_NET_WINDOW_SIZE) / static_cast<float>(out_side - 1);
    }

    std::vector<custom::Face> boxes;

    for (int i = 0; i < size; i++) {
        if (scores_data[i] >= (threshold)) {
            float y = static_cast<float>(i / w);
            float x = static_cast<float>(i - w * y);

            custom::Face faceInfo;
            custom::BBox& faceBox = faceInfo.bbox;

            faceBox.x1 = std::max(0, static_cast<int>((x * stride) / scaleFactor));
            faceBox.y1 = std::max(0, static_cast<int>((y * stride) / scaleFactor));
            faceBox.x2 = static_cast<int>((x * stride + P_NET_WINDOW_SIZE - 1.0f) / scaleFactor);
            faceBox.y2 = static_cast<int>((y * stride + P_NET_WINDOW_SIZE - 1.0f) / scaleFactor);
            faceInfo.regression[0] = reg_data[i];
            faceInfo.regression[1] = reg_data[i + size];
            faceInfo.regression[2] = reg_data[i + 2 * size];
            faceInfo.regression[3] = reg_data[i + 3 * size];
            faceInfo.score = scores_data[i];
            boxes.push_back(faceInfo);
        }
    }

    return boxes;
}
} // anonymous namespace

//Custom kernels implementation
GAPI_OCV_KERNEL(OCVBuildFaces, custom::BuildFaces) {
    static void run(const cv::Mat & in_scores,
                    const cv::Mat & in_regresssions,
                    const float scaleFactor,
                    const float threshold,
                    std::vector<custom::Face> &out_faces) {
        out_faces = buildFaces(in_scores, in_regresssions, scaleFactor, threshold);
    }
}; // GAPI_OCV_KERNEL(BuildFaces)

GAPI_OCV_KERNEL(OCVRunNMS, custom::RunNMS) {
    static void run(const std::vector<custom::Face> &in_faces,
                    const float threshold,
                    const bool useMin,
                    std::vector<custom::Face> &out_faces) {
                    std::vector<custom::Face> in_faces_copy = in_faces;
        out_faces = custom::Face::runNMS(in_faces_copy, threshold, useMin);
    }
}; // GAPI_OCV_KERNEL(RunNMS)

GAPI_OCV_KERNEL(OCVAccumulatePyramidOutputs, custom::AccumulatePyramidOutputs) {
    static void run(const std::vector<custom::Face> &total_faces,
                    const std::vector<custom::Face> &in_faces,
                    std::vector<custom::Face> &out_faces) {
                    out_faces = total_faces;
        out_faces.insert(out_faces.end(), in_faces.begin(), in_faces.end());
    }
}; // GAPI_OCV_KERNEL(AccumulatePyramidOutputs)

GAPI_OCV_KERNEL(OCVApplyRegression, custom::ApplyRegression) {
    static void run(const std::vector<custom::Face> &in_faces,
                    const bool addOne,
                    std::vector<custom::Face> &out_faces) {
        std::vector<custom::Face> in_faces_copy = in_faces;
        custom::Face::applyRegression(in_faces_copy, addOne);
        out_faces.clear();
        out_faces.insert(out_faces.end(), in_faces_copy.begin(), in_faces_copy.end());
    }
}; // GAPI_OCV_KERNEL(ApplyRegression)

GAPI_OCV_KERNEL(OCVBBoxesToSquares, custom::BBoxesToSquares) {
    static void run(const std::vector<custom::Face> &in_faces,
                    std::vector<custom::Face> &out_faces) {
        std::vector<custom::Face> in_faces_copy = in_faces;
        custom::Face::bboxes2Squares(in_faces_copy);
        out_faces.clear();
        out_faces.insert(out_faces.end(), in_faces_copy.begin(), in_faces_copy.end());
    }
}; // GAPI_OCV_KERNEL(BBoxesToSquares)

GAPI_OCV_KERNEL(OCVR_O_NetPreProcGetROIs, custom::R_O_NetPreProcGetROIs) {
    static void run(const std::vector<custom::Face> &in_faces,
                    const cv::Size & in_image_size,
                    std::vector<cv::Rect> &outs) {
        outs.clear();
        for (const auto& face : in_faces) {
            cv::Rect tmp_rect = face.bbox.getRect();
            //Compare to transposed sizes width<->height
            tmp_rect &= cv::Rect(tmp_rect.x, tmp_rect.y, in_image_size.height - tmp_rect.x, in_image_size.width - tmp_rect.y) &
                        cv::Rect(0, 0, in_image_size.height, in_image_size.width);
            outs.push_back(tmp_rect);
        }
    }
}; // GAPI_OCV_KERNEL(R_O_NetPreProcGetROIs)

GAPI_OCV_KERNEL(OCVRNetPostProc, custom::RNetPostProc) {
    static void run(const std::vector<custom::Face> &in_faces,
                    const std::vector<cv::Mat> &in_scores,
                    const std::vector<cv::Mat> &in_regresssions,
                    const float threshold,
                    std::vector<custom::Face> &out_faces) {
        out_faces.clear();
        for (unsigned int k = 0; k < in_faces.size(); ++k) {
            const float* scores_data = in_scores[k].ptr<float>();
            const float* reg_data = in_regresssions[k].ptr<float>();
            if (scores_data[1] >= threshold) {
                custom::Face info = in_faces[k];
                info.score = scores_data[1];
                std::copy_n(reg_data, NUM_REGRESSIONS, info.regression.begin());
                out_faces.push_back(info);
            }
        }
    }
}; // GAPI_OCV_KERNEL(RNetPostProc)

GAPI_OCV_KERNEL(OCVONetPostProc, custom::ONetPostProc) {
    static void run(const std::vector<custom::Face> &in_faces,
                    const std::vector<cv::Mat> &in_scores,
                    const std::vector<cv::Mat> &in_regresssions,
                    const std::vector<cv::Mat> &in_landmarks,
                    const float threshold,
                    std::vector<custom::Face> &out_faces) {
        out_faces.clear();
        for (unsigned int k = 0; k < in_faces.size(); ++k) {
            const float* scores_data = in_scores[k].ptr<float>();
            const float* reg_data = in_regresssions[k].ptr<float>();
            const float* landmark_data = in_landmarks[k].ptr<float>();
            if (scores_data[1] >= threshold) {
                custom::Face info = in_faces[k];
                info.score = scores_data[1];
                for (size_t i = 0; i < 4; ++i) {
                    info.regression[i] = reg_data[i];
                }
                float w = info.bbox.x2 - info.bbox.x1 + 1.0f;
                float h = info.bbox.y2 - info.bbox.y1 + 1.0f;

                for (size_t p = 0; p < NUM_PTS; ++p) {
                    info.ptsCoords[2 * p] =
                        info.bbox.x1 + static_cast<float>(landmark_data[NUM_PTS + p]) * w - 1;
                    info.ptsCoords[2 * p + 1] = info.bbox.y1 + static_cast<float>(landmark_data[p]) * h - 1;
                }

                out_faces.push_back(info);
            }
        }
    }
}; // GAPI_OCV_KERNEL(ONetPostProc)

GAPI_OCV_KERNEL(OCVSwapFaces, custom::SwapFaces) {
    static void run(const std::vector<custom::Face> &in_faces,
                    std::vector<custom::Face> &out_faces) {
        std::vector<custom::Face> in_faces_copy = in_faces;
        out_faces.clear();
        if (!in_faces_copy.empty()) {
            for (size_t i = 0; i < in_faces_copy.size(); ++i) {
                std::swap(in_faces_copy[i].bbox.x1, in_faces_copy[i].bbox.y1);
                std::swap(in_faces_copy[i].bbox.x2, in_faces_copy[i].bbox.y2);
                for (size_t p = 0; p < NUM_PTS; ++p) {
                    std::swap(in_faces_copy[i].ptsCoords[2 * p], in_faces_copy[i].ptsCoords[2 * p + 1]);
                }
            }
            out_faces = in_faces_copy;
        }
    }
}; // GAPI_OCV_KERNEL(SwapFaces)

using rectPoints = std::pair<cv::Rect, std::vector<cv::Point>>;

GAPI_OCV_KERNEL(OCVBoxesAndMarks, custom::BoxesAndMarks) {
    static void run(const cv::Mat& in,
                    const std::vector<custom::Face> &in_faces,
                          std::vector<cv::gapi::wip::draw::Prim>& out_prims) {
        std::vector<rectPoints> data;
        // show the image with faces in it
        for (const auto& out_face : in_faces) {
            std::vector<cv::Point> pts;
            for (size_t p = 0; p < NUM_PTS; ++p) {
                pts.push_back(
                    cv::Point(static_cast<int>(out_face.ptsCoords[2 * p]), static_cast<int>(out_face.ptsCoords[2 * p + 1])));
            }
            const auto rect = out_face.bbox.getRect();
            const auto d = std::make_pair(rect, pts);
            data.push_back(d);
        }

        out_prims.clear();
        const auto rct = [](const cv::Rect &rc) {
            return cv::gapi::wip::draw::Rect(rc, cv::Scalar(0,255,0 ), 1);
        };

         const auto crl = [](const cv::Point &point) {
            return cv::gapi::wip::draw::Circle(point, 3, cv::Scalar(0, 255, 255));
        };

        for (const auto& el : data) {
            out_prims.emplace_back(rct(el.first));
            for (const auto& point : el.second) {
                out_prims.emplace_back(crl(point));
            }
        }
    }
}; // GAPI_OCV_KERNEL(BoxesAndMarks)

cv::gapi::GKernelPackage custom::kernels() {
    return cv::gapi::kernels<OCVBuildFaces,
                             OCVRunNMS,
                             OCVAccumulatePyramidOutputs,
                             OCVApplyRegression,
                             OCVBBoxesToSquares,
                             OCVR_O_NetPreProcGetROIs,
                             OCVRNetPostProc,
                             OCVONetPostProc,
                             OCVSwapFaces,
                             OCVBoxesAndMarks>();
}
