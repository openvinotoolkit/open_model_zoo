// Copyright (C) 2023 KNS Group LLC (YADRO)
// SPDX-License-Identifier: Apache-2.0
//

#include "models.hpp"
#include <opencv2/imgproc.hpp>
#include <vector>

namespace {
    static const float h = 112.;
    static const float w = 96.;
    // reference landmarks points in the unit square [0,1]x[0,1]
    static const float REF_LANDMARKS_NORMED[] = {
        30.2946f / w, 51.6963f / h, 65.5318f / w, 51.5014f / h, 48.0252f / w,
        71.7366f / h, 33.5493f / w, 92.3655f / h, 62.7299f / w, 92.2041f / h
    };
}

cv::Mat getTransform(cv::Mat* src, cv::Mat* dst) {
    cv::Mat colMeanSrc;
    reduce(*src, colMeanSrc, 0, cv::REDUCE_AVG);
    for (int i = 0; i < src->rows; i++) {
        src->row(i) -= colMeanSrc;
    }

    cv::Mat colMeanDst;
    reduce(*dst, colMeanDst, 0, cv::REDUCE_AVG);
    for (int i = 0; i < dst->rows; i++) {
        dst->row(i) -= colMeanDst;
    }

    cv::Scalar mean, devSrc, devDst;
    cv::meanStdDev(*src, mean, devSrc);
    devSrc(0) =
            std::max(static_cast<double>(std::numeric_limits<float>::epsilon()), devSrc(0));
    *src /= devSrc(0);
    cv::meanStdDev(*dst, mean, devDst);
    devDst(0) =
            std::max(static_cast<double>(std::numeric_limits<float>::epsilon()), devDst(0));
    *dst /= devDst(0);

    cv::Mat w, u, vt;
    cv::SVD::compute((*src).t() * (*dst), w, u, vt);
    cv::Mat r = (u * vt).t();
    cv::Mat m(2, 3, CV_32F);
    m.colRange(0, 2) = r * (devDst(0) / devSrc(0));
    m.col(2) = (colMeanDst.t() - m.colRange(0, 2) * colMeanSrc.t());
    return m;
}

void alignFaces(std::vector<cv::Mat>& faceImages, const std::vector<cv::Mat>& landmarksVec) {
    if (landmarksVec.size() == 0) {
        return;
    }
    CV_Assert(faceImages.size() == landmarksVec.size());
    cv::Mat refLandmarks = cv::Mat(5, 2, CV_32F);

    for (size_t j = 0; j < faceImages.size(); j++) {
        auto lms = landmarksVec.at(j).clone();
        for (int i = 0; i < refLandmarks.rows; i++) {
            refLandmarks.at<float>(i, 0) =
                    REF_LANDMARKS_NORMED[2 * i] * faceImages.at(j).cols;
            refLandmarks.at<float>(i, 1) =
                    REF_LANDMARKS_NORMED[2 * i + 1] * faceImages.at(j).rows;
            lms = lms.reshape(1, 5);
            lms.at<float>(i, 0) *= faceImages.at(j).cols;
            lms.at<float>(i, 1) *= faceImages.at(j).rows;
        }
        cv::Mat m = getTransform(&refLandmarks, &lms);
        cv::warpAffine(faceImages.at(j), faceImages.at(j), m,
                       faceImages.at(j).size(), cv::WARP_INVERSE_MAP);
    }
}
