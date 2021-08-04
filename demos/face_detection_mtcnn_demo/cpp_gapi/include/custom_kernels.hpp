// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>

#define NUM_PTS 5
#define NUM_REGRESSIONS 4

namespace custom {

struct BBox {
    int x1;
    int y1;
    int x2;
    int y2;

    cv::Rect getRect() const { return cv::Rect(x1,
                                               y1,
                                               x2 - x1,
                                               y2 - y1); }

    BBox getSquare() const {
        BBox bbox;
        float bboxWidth = static_cast<float>(x2 - x1);
        float bboxHeight = static_cast<float>(y2 - y1);
        float side = std::max(bboxWidth, bboxHeight);
        bbox.x1 = static_cast<int>(static_cast<float>(x1) + (bboxWidth - side) * 0.5f);
        bbox.y1 = static_cast<int>(static_cast<float>(y1) + (bboxHeight - side) * 0.5f);
        bbox.x2 = static_cast<int>(static_cast<float>(bbox.x1) + side);
        bbox.y2 = static_cast<int>(static_cast<float>(bbox.y1) + side);
        return bbox;
    }
};

struct Face {
    BBox bbox;
    float score;
    std::array<float, NUM_REGRESSIONS> regression;
    std::array<float, 2 * NUM_PTS> ptsCoords;

    static void applyRegression(std::vector<Face>& faces, bool addOne = false) {
        for (auto& face : faces) {
            float bboxWidth = face.bbox.x2 - face.bbox.x1 + static_cast<float>(addOne);
            float bboxHeight = face.bbox.y2 - face.bbox.y1 + static_cast<float>(addOne);
            face.bbox.x1 = static_cast<int>(static_cast<float>(face.bbox.x1) + (face.regression[1] * bboxWidth));
            face.bbox.y1 = static_cast<int>(static_cast<float>(face.bbox.y1) + (face.regression[0] * bboxHeight));
            face.bbox.x2 = static_cast<int>(static_cast<float>(face.bbox.x2) + (face.regression[3] * bboxWidth));
            face.bbox.y2 = static_cast<int>(static_cast<float>(face.bbox.y2) + (face.regression[2] * bboxHeight));
        }
    }

    static void bboxes2Squares(std::vector<Face>& faces) {
        for (auto& face : faces) {
            face.bbox = face.bbox.getSquare();
        }
    }

    static std::vector<Face> runNMS(std::vector<Face>& faces, const float threshold,
                                    const bool useMin = false) {
        std::vector<Face> facesNMS;
        if (faces.empty()) {
            return facesNMS;
        }

        std::sort(faces.begin(), faces.end(), [](const Face& f1, const Face& f2) {
            return f1.score > f2.score;
        });

        std::vector<int> indices(faces.size());
        std::iota(indices.begin(), indices.end(), 0);

        while (indices.size() > 0) {
            const int idx = indices[0];
            facesNMS.push_back(faces[idx]);
            std::vector<int> tmpIndices = indices;
            indices.clear();
            const float area1 = static_cast<float>(faces[idx].bbox.x2 - faces[idx].bbox.x1 + 1) *
                static_cast<float>(faces[idx].bbox.y2 - faces[idx].bbox.y1 + 1);
            for (size_t i = 1; i < tmpIndices.size(); ++i) {
                int tmpIdx = tmpIndices[i];
                const float interX1 = static_cast<float>(std::max(faces[idx].bbox.x1, faces[tmpIdx].bbox.x1));
                const float interY1 = static_cast<float>(std::max(faces[idx].bbox.y1, faces[tmpIdx].bbox.y1));
                const float interX2 = static_cast<float>(std::min(faces[idx].bbox.x2, faces[tmpIdx].bbox.x2));
                const float interY2 = static_cast<float>(std::min(faces[idx].bbox.y2, faces[tmpIdx].bbox.y2));

                const float bboxWidth = std::max(0.0f, (interX2 - interX1 + 1));
                const float bboxHeight = std::max(0.0f, (interY2 - interY1 + 1));

                const float interArea = bboxWidth * bboxHeight;
                const float area2 = static_cast<float>(faces[tmpIdx].bbox.x2 - faces[tmpIdx].bbox.x1 + 1) *
                    static_cast<float>(faces[tmpIdx].bbox.y2 - faces[tmpIdx].bbox.y1 + 1);
                float overlap = 0.0;
                if (useMin) {
                    overlap = interArea / std::min(area1, area2);
                } else {
                    overlap = interArea / (area1 + area2 - interArea);
                }
                if (overlap <= threshold) {
                    indices.push_back(tmpIdx);
                }
            }
        }
        return facesNMS;
    }
};

// Define networks for this sample
using GMat2 = std::tuple<cv::GMat, cv::GMat>;
using GMat3 = std::tuple<cv::GMat, cv::GMat, cv::GMat>;
using GMats = cv::GArray<cv::GMat>;
using GRects = cv::GArray<cv::Rect>;
using GSize = cv::GOpaque<cv::Size>;
using GPrims = cv::GArray<cv::gapi::wip::draw::Prim>;
using GFaces = cv::GArray<custom::Face>;
G_API_OP(BuildFaces,
         <GFaces(cv::GMat, cv::GMat, float, float)>,
         "custom.mtcnn.build_faces") {
         static cv::GArrayDesc outMeta(const cv::GMatDesc&,
                                       const cv::GMatDesc&,
                                       const float,
                                       const float) {
              return cv::empty_array_desc();
    }
};

G_API_OP(RunNMS,
         <GFaces(GFaces, float, bool)>,
         "custom.mtcnn.run_nms") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
                                       const float, const bool) {
             return cv::empty_array_desc();
    }
};

G_API_OP(AccumulatePyramidOutputs,
         <GFaces(GFaces, GFaces)>,
         "custom.mtcnn.accumulate_pyramid_outputs") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
                                       const cv::GArrayDesc&) {
             return cv::empty_array_desc();
    }
};

G_API_OP(ApplyRegression,
         <GFaces(GFaces, bool)>,
         "custom.mtcnn.apply_regression") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&, const bool) {
             return cv::empty_array_desc();
    }
};

G_API_OP(BBoxesToSquares,
         <GFaces(GFaces)>,
         "custom.mtcnn.bboxes_to_squares") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&) {
              return cv::empty_array_desc();
    }
};

G_API_OP(R_O_NetPreProcGetROIs,
         <GRects(GFaces, GSize)>,
         "custom.mtcnn.bboxes_r_o_net_preproc_get_rois") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&, const cv::GOpaqueDesc&) {
              return cv::empty_array_desc();
    }
};

G_API_OP(RNetPostProc,
         <GFaces(GFaces, GMats, GMats, float)>,
         "custom.mtcnn.rnet_postproc") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
                                       const cv::GArrayDesc&,
                                       const cv::GArrayDesc&,
                                       const float) {
             return cv::empty_array_desc();
    }
};

G_API_OP(ONetPostProc,
         <GFaces(GFaces, GMats, GMats, GMats, float)>,
         "custom.mtcnn.onet_postproc") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
                                       const cv::GArrayDesc&,
                                       const cv::GArrayDesc&,
                                       const cv::GArrayDesc&,
                                       const float) {
             return cv::empty_array_desc();
    }
};

G_API_OP(SwapFaces,
         <GFaces(GFaces)>,
         "custom.mtcnn.swap_faces") {
         static cv::GArrayDesc outMeta(const cv::GArrayDesc&) {
              return cv::empty_array_desc();
    }
};

G_API_OP(BoxesAndMarks,
         <GPrims(const cv::GMat,
                 const cv::GArray<Face>)>,
         "custom.boxes_and_markss") {
        static cv::GArrayDesc outMeta(const cv::GMatDesc&,
                                      const cv::GArrayDesc&) {
            return cv::empty_array_desc();
        }
    };

cv::gapi::GKernelPackage kernels();
} //namespace custom
